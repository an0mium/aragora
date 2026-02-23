"""
Tests for Organization Management Handler (aragora/server/handlers/organizations.py).

Covers all 14 endpoints:
- GET    /api/org/{org_id}                         - Get organization details
- PUT    /api/org/{org_id}                         - Update organization settings
- GET    /api/org/{org_id}/members                 - List organization members
- POST   /api/org/{org_id}/invite                  - Invite user to organization
- GET    /api/org/{org_id}/invitations              - List pending invitations
- DELETE /api/org/{org_id}/invitations/{inv_id}     - Revoke invitation
- DELETE /api/org/{org_id}/members/{user_id}        - Remove member
- PUT    /api/org/{org_id}/members/{user_id}/role   - Update member role
- GET    /api/invitations/pending                   - List pending invitations for user
- POST   /api/invitations/{token}/accept            - Accept invitation
- GET    /api/user/organizations                    - List user organizations
- POST   /api/user/organizations/switch             - Switch active organization
- POST   /api/user/organizations/default            - Set default organization
- DELETE /api/user/organizations/{org_id}           - Leave organization

Test categories:
- can_handle() routing (matching and non-matching paths)
- Happy path for each endpoint
- Method not allowed (405) for wrong HTTP methods
- Authentication errors (401, 403)
- Validation errors (400): missing fields, invalid JSON, bad roles
- Not found errors (404)
- Service unavailable (503): missing user_store
- Server errors (500): failed store operations
- Rate limiting (429)
- RBAC permission checks
- Edge cases: owner leave prevention, self-removal, role hierarchy
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.organizations import (
    MAX_SETTINGS_KEYS,
    MAX_SETTINGS_VALUE_SIZE,
    OrganizationsHandler,
    ROLE_HIERARCHY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock data models
# ---------------------------------------------------------------------------


@dataclass
class MockTierLimits:
    """Mock tier limits."""

    debates_per_month: int = 100
    users_per_org: int = 10
    api_access: bool = True
    all_agents: bool = True
    custom_agents: bool = False
    sso_enabled: bool = False
    audit_logs: bool = True
    priority_support: bool = False
    price_monthly_cents: int = 2900

    def to_dict(self) -> dict:
        return {
            "debates_per_month": self.debates_per_month,
            "users_per_org": self.users_per_org,
            "api_access": self.api_access,
        }


class MockSubscriptionTier:
    """Mock subscription tier enum value."""

    def __init__(self, value: str = "professional"):
        self.value = value


@dataclass
class MockOrganization:
    """Mock organization model."""

    id: str = "org-001"
    name: str = "Test Org"
    slug: str = "test-org"
    tier: Any = field(default_factory=lambda: MockSubscriptionTier("professional"))
    owner_id: str = "user-owner"
    created_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    debates_used_this_month: int = 5
    limits: MockTierLimits = field(default_factory=MockTierLimits)
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "tier": self.tier.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class MockUser:
    """Mock user model."""

    id: str = "user-001"
    email: str = "admin@example.com"
    name: str = "Admin User"
    org_id: str | None = "org-001"
    role: str = "admin"
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    last_login_at: datetime | None = None


@dataclass
class MockAuthContext:
    """Mock auth context from extract_user_from_request."""

    is_authenticated: bool = True
    user_id: str = "user-001"


@dataclass
class MockInvitation:
    """Mock organization invitation."""

    id: str = "inv-001"
    org_id: str = "org-001"
    email: str = "invitee@example.com"
    role: str = "member"
    token: str = "test-invite-token"
    invited_by: str = "user-001"
    status: str = "pending"
    is_pending: bool = True
    is_expired: bool = False
    created_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=7)
    )
    accepted_at: datetime | None = None

    def to_dict(self, include_token: bool = False) -> dict:
        data = {
            "id": self.id,
            "org_id": self.org_id,
            "email": self.email,
            "role": self.role,
            "invited_by": self.invited_by,
            "status": self.status,
            "is_pending": self.is_pending,
            "is_expired": self.is_expired,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
        }
        if include_token:
            data["token"] = self.token
        return data


# ---------------------------------------------------------------------------
# Mock HTTP handler (simulates HTTP request)
# ---------------------------------------------------------------------------


class MockHTTPHandler:
    """Mock HTTP request handler for testing."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.headers = {"Content-Length": str(len(body_bytes))}
            self.rfile = MagicMock()
            self.rfile.read.return_value = body_bytes
        else:
            self.headers = {"Content-Length": "0"}
            self.rfile = MagicMock()
            self.rfile.read.return_value = b""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_user_store():
    """Create a mock user store with standard methods."""
    store = MagicMock()
    org = MockOrganization()
    user = MockUser()

    store.get_user_by_id.return_value = user
    store.get_organization_by_id.return_value = org
    store.get_org_members.return_value = [user]
    store.get_user_by_email.return_value = None
    store.add_user_to_org.return_value = True
    store.remove_user_from_org.return_value = True
    store.update_organization.return_value = True
    store.update_user.return_value = True
    store.create_invitation.return_value = True
    store.get_invitation_by_email.return_value = None
    store.get_invitation_by_token.return_value = None
    store.get_invitation_by_id.return_value = None
    store.get_invitations_for_org.return_value = []
    store.get_pending_invitations_by_email.return_value = []
    store.update_invitation_status.return_value = True
    store.cleanup_expired_invitations.return_value = None
    return store


@pytest.fixture
def handler(mock_user_store):
    """Create an OrganizationsHandler with a mock user store."""
    return OrganizationsHandler(ctx={"user_store": mock_user_store})


@pytest.fixture
def handler_no_store():
    """Create an OrganizationsHandler without a user store."""
    return OrganizationsHandler(ctx={})


@pytest.fixture
def admin_user():
    """Admin user (role=admin, org_id=org-001)."""
    return MockUser(id="user-001", role="admin", org_id="org-001")


@pytest.fixture
def owner_user():
    """Owner user (role=owner, org_id=org-001)."""
    return MockUser(id="user-owner", role="owner", org_id="org-001")


@pytest.fixture
def member_user():
    """Member user (role=member, org_id=org-001)."""
    return MockUser(id="user-member", role="member", org_id="org-001")


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter between tests to avoid cross-test pollution."""
    from aragora.server.handlers.organizations import _org_limiter

    _org_limiter._requests.clear()
    yield


@pytest.fixture(autouse=True)
def patch_audit(monkeypatch):
    """Patch audit functions to no-op."""
    monkeypatch.setattr(
        "aragora.server.handlers.organizations.audit_admin", lambda **kw: None
    )
    monkeypatch.setattr(
        "aragora.server.handlers.organizations.audit_data", lambda **kw: None
    )


@pytest.fixture(autouse=True)
def patch_rbac_check(monkeypatch):
    """Bypass inline _check_rbac_permission to always allow."""
    monkeypatch.setattr(
        OrganizationsHandler,
        "_check_rbac_permission",
        lambda self, handler, perm, user=None: None,
    )


@pytest.fixture(autouse=True)
def patch_extract_user(monkeypatch):
    """Patch extract_user_from_request to return a mock auth context."""
    mock_auth = MockAuthContext(is_authenticated=True, user_id="user-001")
    monkeypatch.setattr(
        "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
        lambda self, handler: (MockUser(), mock_auth),
    )
    yield mock_auth


def _patch_user(monkeypatch, user: MockUser | None, authenticated: bool = True):
    """Override _get_current_user to return a specific user."""
    auth = MockAuthContext(is_authenticated=authenticated, user_id=user.id if user else None)
    monkeypatch.setattr(
        "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
        lambda self, handler: (user, auth) if authenticated else (None, None),
    )


def _patch_user_none(monkeypatch):
    """Override _get_current_user to return no user (unauthenticated)."""
    monkeypatch.setattr(
        "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
        lambda self, handler: (None, None),
    )


# ===========================================================================
# can_handle() tests
# ===========================================================================


class TestCanHandle:
    """Test can_handle() route matching."""

    def test_org_detail(self, handler):
        assert handler.can_handle("/api/org/org-001") is True

    def test_org_detail_v1(self, handler):
        assert handler.can_handle("/api/v1/org/org-001") is True

    def test_org_members(self, handler):
        assert handler.can_handle("/api/org/org-001/members") is True

    def test_org_members_v1(self, handler):
        assert handler.can_handle("/api/v1/org/org-001/members") is True

    def test_org_invite(self, handler):
        assert handler.can_handle("/api/org/org-001/invite") is True

    def test_org_invitations(self, handler):
        assert handler.can_handle("/api/org/org-001/invitations") is True

    def test_org_invitation_detail(self, handler):
        assert handler.can_handle("/api/org/org-001/invitations/inv-001") is True

    def test_org_member_detail(self, handler):
        assert handler.can_handle("/api/org/org-001/members/user-002") is True

    def test_org_member_role(self, handler):
        assert handler.can_handle("/api/org/org-001/members/user-002/role") is True

    def test_user_organizations(self, handler):
        assert handler.can_handle("/api/user/organizations") is True

    def test_user_organizations_switch(self, handler):
        assert handler.can_handle("/api/user/organizations/switch") is True

    def test_user_organizations_default(self, handler):
        assert handler.can_handle("/api/user/organizations/default") is True

    def test_user_org_leave(self, handler):
        assert handler.can_handle("/api/user/organizations/org-001") is True

    def test_invitations_pending(self, handler):
        assert handler.can_handle("/api/invitations/pending") is True

    def test_invitations_accept(self, handler):
        assert handler.can_handle("/api/invitations/test-token/accept") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_partial_path(self, handler):
        assert handler.can_handle("/api/org") is False

    def test_v1_user_organizations(self, handler):
        assert handler.can_handle("/api/v1/user/organizations") is True

    def test_v1_invitations_pending(self, handler):
        assert handler.can_handle("/api/v1/invitations/pending") is True

    def test_v1_accept_invitation(self, handler):
        assert handler.can_handle("/api/v1/invitations/abc-123/accept") is True

    def test_org_id_with_underscores(self, handler):
        assert handler.can_handle("/api/org/my_org_123") is True

    def test_org_id_with_hyphens(self, handler):
        assert handler.can_handle("/api/org/my-org-123") is True


# ===========================================================================
# GET /api/org/{org_id} - Get organization details
# ===========================================================================


class TestGetOrganization:
    """Tests for _get_organization endpoint."""

    def test_success(self, handler, mock_user_store):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001", {}, h, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert "organization" in body
        assert body["organization"]["id"] == "org-001"
        assert body["organization"]["name"] == "Test Org"

    def test_returns_member_count(self, handler, mock_user_store):
        mock_user_store.get_org_members.return_value = [MockUser(), MockUser()]
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001", {}, h, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["organization"]["member_count"] == 2

    def test_org_not_found(self, handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = None
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001", {}, h, method="GET")
        assert _status(result) == 404

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001", {}, h, method="GET")
        assert _status(result) == 401

    def test_wrong_org(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(org_id="org-999"))
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001", {}, h, method="GET")
        assert _status(result) == 403

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle("/api/org/org-001", {}, h, method="DELETE")
        assert _status(result) == 405

    def test_v1_path(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/v1/org/org-001", {}, h, method="GET")
        assert _status(result) == 200

    def test_returns_settings(self, handler, mock_user_store):
        org = MockOrganization(settings={"theme": "dark"})
        mock_user_store.get_organization_by_id.return_value = org
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001", {}, h, method="GET")
        assert _body(result)["organization"]["settings"] == {"theme": "dark"}


# ===========================================================================
# PUT /api/org/{org_id} - Update organization
# ===========================================================================


class TestUpdateOrganization:
    """Tests for _update_organization endpoint."""

    def test_update_name(self, handler, mock_user_store):
        h = MockHTTPHandler(body={"name": "New Name"}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 200
        body = _body(result)
        assert body["message"] == "Organization updated"

    def test_update_settings(self, handler, mock_user_store):
        h = MockHTTPHandler(body={"settings": {"theme": "dark"}}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 200

    def test_update_name_and_settings(self, handler, mock_user_store):
        h = MockHTTPHandler(
            body={"name": "New Name", "settings": {"key": "val"}}, method="PUT"
        )
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 200

    def test_name_too_short(self, handler):
        h = MockHTTPHandler(body={"name": "X"}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 400
        assert "at least 2" in _body(result).get("error", "")

    def test_name_too_long(self, handler):
        h = MockHTTPHandler(body={"name": "X" * 101}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 400
        assert "at most 100" in _body(result).get("error", "")

    def test_no_valid_fields(self, handler):
        h = MockHTTPHandler(body={"unknown": "value"}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 400
        assert "No valid fields" in _body(result).get("error", "")

    def test_invalid_json(self, handler):
        h = MockHTTPHandler(method="PUT")
        h.headers = {"Content-Length": "5"}
        h.rfile.read.return_value = b"notjs"
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 400

    def test_requires_admin_role(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(role="member", org_id="org-001"))
        h = MockHTTPHandler(body={"name": "New"}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 403

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(body={"name": "New"}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 401

    def test_store_failure(self, handler, mock_user_store):
        mock_user_store.update_organization.return_value = False
        h = MockHTTPHandler(body={"name": "New Name"}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 500

    def test_too_many_settings_keys(self, handler):
        settings = {f"key_{i}": "val" for i in range(MAX_SETTINGS_KEYS + 1)}
        h = MockHTTPHandler(body={"settings": settings}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 400
        assert "Too many settings" in _body(result).get("error", "")

    def test_settings_value_too_large(self, handler):
        settings = {"big_key": "x" * (MAX_SETTINGS_VALUE_SIZE + 1)}
        h = MockHTTPHandler(body={"settings": settings}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 400
        assert "too large" in _body(result).get("error", "")

    def test_empty_body(self, handler):
        h = MockHTTPHandler(body={}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 400


# ===========================================================================
# GET /api/org/{org_id}/members - List members
# ===========================================================================


class TestListMembers:
    """Tests for _list_members endpoint."""

    def test_success(self, handler, mock_user_store):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001/members", {}, h, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert "members" in body
        assert body["count"] == 1

    def test_multiple_members(self, handler, mock_user_store):
        mock_user_store.get_org_members.return_value = [
            MockUser(id="u1", email="a@example.com", name="A"),
            MockUser(id="u2", email="b@example.com", name="B"),
        ]
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001/members", {}, h, method="GET")
        body = _body(result)
        assert body["count"] == 2
        assert len(body["members"]) == 2

    def test_member_fields(self, handler, mock_user_store):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001/members", {}, h, method="GET")
        member = _body(result)["members"][0]
        assert "id" in member
        assert "email" in member
        assert "name" in member
        assert "role" in member
        assert "is_active" in member

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001/members", {}, h, method="GET")
        assert _status(result) == 401

    def test_wrong_org(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(org_id="other-org"))
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001/members", {}, h, method="GET")
        assert _status(result) == 403

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="POST")
        result = handler.handle("/api/org/org-001/members", {}, h, method="POST")
        assert _status(result) == 405


# ===========================================================================
# POST /api/org/{org_id}/invite - Invite member
# ===========================================================================


class TestInviteMember:
    """Tests for _invite_member endpoint."""

    def test_invite_new_user(self, handler, mock_user_store):
        h = MockHTTPHandler(
            body={"email": "newuser@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 201
        body = _body(result)
        assert "invitation_id" in body
        assert "invite_link" in body

    def test_invite_existing_user_no_org(self, handler, mock_user_store):
        existing = MockUser(id="user-existing", email="existing@example.com", org_id=None)
        mock_user_store.get_user_by_email.return_value = existing
        h = MockHTTPHandler(
            body={"email": "existing@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 200
        body = _body(result)
        assert body["user_id"] == "user-existing"

    def test_invite_existing_member(self, handler, mock_user_store):
        existing = MockUser(id="user-existing", email="member@example.com", org_id="org-001")
        mock_user_store.get_user_by_email.return_value = existing
        h = MockHTTPHandler(
            body={"email": "member@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 400
        assert "already a member" in _body(result).get("error", "")

    def test_invite_user_in_other_org(self, handler, mock_user_store):
        existing = MockUser(id="user-other", email="other@example.com", org_id="org-999")
        mock_user_store.get_user_by_email.return_value = existing
        h = MockHTTPHandler(
            body={"email": "other@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 400
        assert "another organization" in _body(result).get("error", "")

    def test_invite_missing_email(self, handler):
        h = MockHTTPHandler(body={"role": "member"}, method="POST")
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 400

    def test_invite_invalid_role(self, handler):
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "superadmin"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 400

    def test_invite_org_not_found(self, handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = None
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 404

    def test_invite_member_limit_reached(self, handler, mock_user_store):
        org = MockOrganization(limits=MockTierLimits(users_per_org=1))
        mock_user_store.get_organization_by_id.return_value = org
        mock_user_store.get_org_members.return_value = [MockUser()]
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 403
        assert "limit reached" in _body(result).get("error", "")

    def test_invite_requires_admin(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(role="member", org_id="org-001"))
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 403

    def test_invite_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 401

    def test_invite_invalid_json(self, handler):
        h = MockHTTPHandler(method="POST")
        h.headers = {"Content-Length": "3"}
        h.rfile.read.return_value = b"bad"
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 400

    def test_invite_duplicate_pending(self, handler, mock_user_store):
        mock_user_store.get_invitation_by_email.return_value = MockInvitation(
            email="test@example.com", is_pending=True
        )
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 400
        assert "already been sent" in _body(result).get("error", "")

    def test_invite_add_existing_user_failure(self, handler, mock_user_store):
        existing = MockUser(id="user-existing", email="test@example.com", org_id=None)
        mock_user_store.get_user_by_email.return_value = existing
        mock_user_store.add_user_to_org.return_value = False
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "member"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 500

    def test_invite_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001/invite", {}, h, method="GET")
        assert _status(result) == 405

    def test_invite_default_role_is_member(self, handler, mock_user_store):
        h = MockHTTPHandler(body={"email": "test@example.com"}, method="POST")
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 201

    def test_invite_admin_role(self, handler, mock_user_store):
        h = MockHTTPHandler(
            body={"email": "test@example.com", "role": "admin"}, method="POST"
        )
        result = handler.handle("/api/org/org-001/invite", {}, h, method="POST")
        assert _status(result) == 201


# ===========================================================================
# DELETE /api/org/{org_id}/members/{user_id} - Remove member
# ===========================================================================


class TestRemoveMember:
    """Tests for _remove_member endpoint."""

    def test_success(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-target", role="member", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-001", role="admin", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-target", {}, h, method="DELETE"
        )
        assert _status(result) == 200
        assert _body(result)["user_id"] == "user-target"

    def test_remove_owner_forbidden(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-owner", role="owner", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-001", role="admin", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-owner", {}, h, method="DELETE"
        )
        assert _status(result) == 403
        assert "owner" in _body(result).get("error", "").lower()

    def test_remove_self_forbidden(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-001", role="admin", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-001", role="admin", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-001", {}, h, method="DELETE"
        )
        assert _status(result) == 400
        assert "yourself" in _body(result).get("error", "").lower()

    def test_admin_removing_admin_requires_owner(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-admin2", role="admin", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-001", role="admin", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-admin2", {}, h, method="DELETE"
        )
        assert _status(result) == 403
        assert "owner" in _body(result).get("error", "").lower()

    def test_owner_can_remove_admin(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-admin2", role="admin", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-admin2", {}, h, method="DELETE"
        )
        assert _status(result) == 200

    def test_target_user_not_found(self, handler, mock_user_store):
        mock_user_store.get_user_by_id.return_value = None
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/nonexistent", {}, h, method="DELETE"
        )
        assert _status(result) == 404

    def test_target_not_in_org(self, handler, mock_user_store):
        target = MockUser(id="user-other", org_id="org-999")
        mock_user_store.get_user_by_id.return_value = target
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-other", {}, h, method="DELETE"
        )
        assert _status(result) == 400

    def test_store_failure(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-target", role="member", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        mock_user_store.remove_user_from_org.return_value = False
        _patch_user(monkeypatch, MockUser(id="user-001", role="admin", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-target", {}, h, method="DELETE"
        )
        assert _status(result) == 500

    def test_requires_admin(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(role="member", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/members/user-target", {}, h, method="DELETE"
        )
        assert _status(result) == 403

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/org/org-001/members/user-target", {}, h, method="GET"
        )
        assert _status(result) == 405


# ===========================================================================
# PUT /api/org/{org_id}/members/{user_id}/role - Update member role
# ===========================================================================


class TestUpdateMemberRole:
    """Tests for _update_member_role endpoint."""

    def test_success(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-target", role="member", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "admin"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["role"] == "admin"

    def test_demote_admin_to_member(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-target", role="admin", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "member"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 200

    def test_cannot_change_owner_role(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-owner-target", role="owner", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "admin"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-owner-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 403

    def test_invalid_role(self, handler, mock_user_store, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "superadmin"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 400

    def test_empty_role(self, handler, mock_user_store, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": ""}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 400

    def test_requires_owner_role(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-admin", role="admin", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "admin"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 403

    def test_target_not_found(self, handler, mock_user_store, monkeypatch):
        mock_user_store.get_user_by_id.return_value = None
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "admin"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 404

    def test_target_not_in_org(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-other", org_id="org-999")
        mock_user_store.get_user_by_id.return_value = target
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "admin"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-other/role", {}, h, method="PUT"
        )
        assert _status(result) == 400

    def test_store_failure(self, handler, mock_user_store, monkeypatch):
        target = MockUser(id="user-target", role="member", org_id="org-001")
        mock_user_store.get_user_by_id.return_value = target
        mock_user_store.update_user.return_value = False
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(body={"role": "admin"}, method="PUT")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 500

    def test_invalid_json(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(method="PUT")
        h.headers = {"Content-Length": "5"}
        h.rfile.read.return_value = b"notjs"
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="PUT"
        )
        assert _status(result) == 400

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/org/org-001/members/user-target/role", {}, h, method="GET"
        )
        assert _status(result) == 405


# ===========================================================================
# GET /api/user/organizations - List user organizations
# ===========================================================================


class TestListUserOrganizations:
    """Tests for _list_user_organizations endpoint."""

    def test_success(self, handler, mock_user_store):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/user/organizations", {}, h, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert "organizations" in body
        assert len(body["organizations"]) == 1

    def test_no_org(self, handler, monkeypatch, mock_user_store):
        _patch_user(monkeypatch, MockUser(org_id=None))
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/user/organizations", {}, h, method="GET")
        assert _status(result) == 200
        assert _body(result)["organizations"] == []

    def test_org_not_found_in_store(self, handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = None
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/user/organizations", {}, h, method="GET")
        assert _status(result) == 200
        assert _body(result)["organizations"] == []

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/user/organizations", {}, h, method="GET")
        assert _status(result) == 401

    def test_no_store(self, handler_no_store, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
            lambda self, handler: (MockUser(), MockAuthContext()),
        )
        h = MockHTTPHandler(method="GET")
        result = handler_no_store.handle("/api/user/organizations", {}, h, method="GET")
        assert _status(result) == 503

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="PUT")
        result = handler.handle("/api/user/organizations", {}, h, method="PUT")
        assert _status(result) == 405

    def test_response_fields(self, handler, mock_user_store):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/user/organizations", {}, h, method="GET")
        org_data = _body(result)["organizations"][0]
        assert "user_id" in org_data
        assert "org_id" in org_data
        assert "organization" in org_data
        assert "role" in org_data
        assert "is_default" in org_data


# ===========================================================================
# POST /api/user/organizations/switch - Switch organization
# ===========================================================================


class TestSwitchOrganization:
    """Tests for _switch_user_organization endpoint."""

    def test_success(self, handler, mock_user_store):
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 200
        assert "organization" in _body(result)

    def test_missing_org_id(self, handler):
        h = MockHTTPHandler(body={}, method="POST")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_empty_org_id(self, handler):
        h = MockHTTPHandler(body={"org_id": ""}, method="POST")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_not_member(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(org_id="org-999"))
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 403

    def test_org_not_found(self, handler, mock_user_store):
        mock_user_store.get_organization_by_id.return_value = None
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 404

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 401

    def test_invalid_json(self, handler):
        h = MockHTTPHandler(method="POST")
        h.headers = {"Content-Length": "3"}
        h.rfile.read.return_value = b"bad"
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_no_store(self, handler_no_store, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
            lambda self, handler: (MockUser(), MockAuthContext()),
        )
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler_no_store.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 503

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="GET"
        )
        assert _status(result) == 405


# ===========================================================================
# POST /api/user/organizations/default - Set default organization
# ===========================================================================


class TestSetDefaultOrganization:
    """Tests for _set_default_organization endpoint."""

    def test_success(self, handler, mock_user_store):
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler.handle(
            "/api/user/organizations/default", {}, h, method="POST"
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    def test_missing_org_id(self, handler):
        h = MockHTTPHandler(body={}, method="POST")
        result = handler.handle(
            "/api/user/organizations/default", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_not_member(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(org_id="org-999"))
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler.handle(
            "/api/user/organizations/default", {}, h, method="POST"
        )
        assert _status(result) == 403

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler.handle(
            "/api/user/organizations/default", {}, h, method="POST"
        )
        assert _status(result) == 401

    def test_invalid_json(self, handler):
        h = MockHTTPHandler(method="POST")
        h.headers = {"Content-Length": "3"}
        h.rfile.read.return_value = b"bad"
        result = handler.handle(
            "/api/user/organizations/default", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_no_store(self, handler_no_store, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
            lambda self, handler: (MockUser(), MockAuthContext()),
        )
        h = MockHTTPHandler(body={"org_id": "org-001"}, method="POST")
        result = handler_no_store.handle(
            "/api/user/organizations/default", {}, h, method="POST"
        )
        assert _status(result) == 503

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/user/organizations/default", {}, h, method="GET"
        )
        assert _status(result) == 405


# ===========================================================================
# DELETE /api/user/organizations/{org_id} - Leave organization
# ===========================================================================


class TestLeaveOrganization:
    """Tests for _leave_organization endpoint."""

    def test_success(self, handler, mock_user_store, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-001", role="member", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/user/organizations/org-001", {}, h, method="DELETE"
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    def test_owner_cannot_leave(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-owner", role="owner", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/user/organizations/org-001", {}, h, method="DELETE"
        )
        assert _status(result) == 400
        assert "owner" in _body(result).get("error", "").lower()

    def test_not_member(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(org_id="org-999"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/user/organizations/org-001", {}, h, method="DELETE"
        )
        assert _status(result) == 403

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/user/organizations/org-001", {}, h, method="DELETE"
        )
        assert _status(result) == 401

    def test_store_failure(self, handler, mock_user_store, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-001", role="member", org_id="org-001"))
        mock_user_store.remove_user_from_org.return_value = False
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/user/organizations/org-001", {}, h, method="DELETE"
        )
        assert _status(result) == 500

    def test_no_store(self, handler_no_store, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
            lambda self, handler: (MockUser(role="member"), MockAuthContext()),
        )
        h = MockHTTPHandler(method="DELETE")
        result = handler_no_store.handle(
            "/api/user/organizations/org-001", {}, h, method="DELETE"
        )
        assert _status(result) == 503

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/user/organizations/org-001", {}, h, method="GET"
        )
        assert _status(result) == 405

    def test_admin_can_leave(self, handler, mock_user_store, monkeypatch):
        _patch_user(monkeypatch, MockUser(id="user-admin", role="admin", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/user/organizations/org-001", {}, h, method="DELETE"
        )
        assert _status(result) == 200


# ===========================================================================
# GET /api/org/{org_id}/invitations - List invitations
# ===========================================================================


class TestListInvitations:
    """Tests for _list_invitations endpoint."""

    def test_success(self, handler, mock_user_store):
        mock_user_store.get_invitations_for_org.return_value = [
            MockInvitation(id="inv-1", is_pending=True),
            MockInvitation(id="inv-2", is_pending=False, status="accepted"),
        ]
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/org/org-001/invitations", {}, h, method="GET"
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert body["pending_count"] == 1

    def test_empty(self, handler, mock_user_store):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/org/org-001/invitations", {}, h, method="GET"
        )
        assert _status(result) == 200
        assert _body(result)["count"] == 0

    def test_requires_admin(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(role="member", org_id="org-001"))
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/org/org-001/invitations", {}, h, method="GET"
        )
        assert _status(result) == 403

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/org/org-001/invitations", {}, h, method="GET"
        )
        assert _status(result) == 401

    def test_no_store(self, handler_no_store, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
            lambda self, handler: (MockUser(role="admin"), MockAuthContext()),
        )
        h = MockHTTPHandler(method="GET")
        result = handler_no_store.handle(
            "/api/org/org-001/invitations", {}, h, method="GET"
        )
        assert _status(result) == 503

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/org/org-001/invitations", {}, h, method="POST"
        )
        assert _status(result) == 405


# ===========================================================================
# DELETE /api/org/{org_id}/invitations/{invitation_id} - Revoke invitation
# ===========================================================================


class TestRevokeInvitation:
    """Tests for _revoke_invitation endpoint."""

    def test_success(self, handler, mock_user_store):
        inv = MockInvitation(id="inv-001", org_id="org-001", is_pending=True)
        mock_user_store.get_invitation_by_id.return_value = inv
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/invitations/inv-001", {}, h, method="DELETE"
        )
        assert _status(result) == 200
        assert _body(result)["invitation_id"] == "inv-001"

    def test_invitation_not_found(self, handler, mock_user_store):
        mock_user_store.get_invitation_by_id.return_value = None
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/invitations/inv-999", {}, h, method="DELETE"
        )
        assert _status(result) == 404

    def test_invitation_wrong_org(self, handler, mock_user_store):
        inv = MockInvitation(id="inv-001", org_id="org-999")
        mock_user_store.get_invitation_by_id.return_value = inv
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/invitations/inv-001", {}, h, method="DELETE"
        )
        assert _status(result) == 404

    def test_invitation_not_pending(self, handler, mock_user_store):
        inv = MockInvitation(
            id="inv-001", org_id="org-001", is_pending=False, status="accepted"
        )
        mock_user_store.get_invitation_by_id.return_value = inv
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/invitations/inv-001", {}, h, method="DELETE"
        )
        assert _status(result) == 400
        assert "accepted" in _body(result).get("error", "")

    def test_requires_admin(self, handler, monkeypatch):
        _patch_user(monkeypatch, MockUser(role="member", org_id="org-001"))
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/invitations/inv-001", {}, h, method="DELETE"
        )
        assert _status(result) == 403

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="DELETE")
        result = handler.handle(
            "/api/org/org-001/invitations/inv-001", {}, h, method="DELETE"
        )
        assert _status(result) == 401

    def test_no_store(self, handler_no_store, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
            lambda self, handler: (MockUser(role="admin"), MockAuthContext()),
        )
        h = MockHTTPHandler(method="DELETE")
        result = handler_no_store.handle(
            "/api/org/org-001/invitations/inv-001", {}, h, method="DELETE"
        )
        assert _status(result) == 503

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/org/org-001/invitations/inv-001", {}, h, method="GET"
        )
        assert _status(result) == 405


# ===========================================================================
# GET /api/invitations/pending - Get pending invitations for user
# ===========================================================================


class TestGetPendingInvitations:
    """Tests for _get_pending_invitations endpoint."""

    def test_success(self, handler, mock_user_store):
        mock_user_store.get_pending_invitations_by_email.return_value = [
            MockInvitation(id="inv-1", org_id="org-001"),
        ]
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/invitations/pending", {}, h, method="GET")
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert "org_name" in body["invitations"][0]

    def test_empty(self, handler, mock_user_store):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/invitations/pending", {}, h, method="GET")
        assert _status(result) == 200
        assert _body(result)["count"] == 0

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/invitations/pending", {}, h, method="GET")
        assert _status(result) == 401

    def test_org_name_unknown(self, handler, mock_user_store):
        mock_user_store.get_pending_invitations_by_email.return_value = [
            MockInvitation(id="inv-1", org_id="org-missing"),
        ]
        mock_user_store.get_organization_by_id.return_value = None
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/invitations/pending", {}, h, method="GET")
        assert _status(result) == 200
        assert _body(result)["invitations"][0]["org_name"] == "Unknown Organization"

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="POST")
        result = handler.handle("/api/invitations/pending", {}, h, method="POST")
        assert _status(result) == 405


# ===========================================================================
# POST /api/invitations/{token}/accept - Accept invitation
# ===========================================================================


class TestAcceptInvitation:
    """Tests for _accept_invitation endpoint."""

    def test_success(self, handler, mock_user_store, monkeypatch):
        inv = MockInvitation(
            token="abc-token",
            org_id="org-001",
            email="admin@example.com",
            role="member",
            is_pending=True,
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        _patch_user(monkeypatch, MockUser(id="user-001", email="admin@example.com", org_id=None))
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 200
        body = _body(result)
        assert "organization" in body
        assert body["role"] == "member"

    def test_invitation_not_found(self, handler):
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/bad-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 404

    def test_invitation_not_pending(self, handler, mock_user_store):
        inv = MockInvitation(
            token="abc-token", is_pending=False, status="accepted",
            email="admin@example.com",
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_wrong_email(self, handler, mock_user_store, monkeypatch):
        inv = MockInvitation(
            token="abc-token", email="other@example.com", is_pending=True
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        _patch_user(monkeypatch, MockUser(email="admin@example.com", org_id=None))
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 403

    def test_already_in_org(self, handler, mock_user_store):
        inv = MockInvitation(
            token="abc-token", email="admin@example.com", is_pending=True
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 400
        assert "already a member" in _body(result).get("error", "")

    def test_org_no_longer_exists(self, handler, mock_user_store, monkeypatch):
        inv = MockInvitation(
            token="abc-token", email="admin@example.com", is_pending=True
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        mock_user_store.get_organization_by_id.return_value = None
        _patch_user(monkeypatch, MockUser(email="admin@example.com", org_id=None))
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 404

    def test_org_member_limit(self, handler, mock_user_store, monkeypatch):
        inv = MockInvitation(
            token="abc-token", email="admin@example.com",
            org_id="org-001", is_pending=True,
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        org = MockOrganization(limits=MockTierLimits(users_per_org=1))
        mock_user_store.get_organization_by_id.return_value = org
        mock_user_store.get_org_members.return_value = [MockUser()]
        _patch_user(monkeypatch, MockUser(email="admin@example.com", org_id=None))
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 403

    def test_add_user_failure(self, handler, mock_user_store, monkeypatch):
        inv = MockInvitation(
            token="abc-token", email="admin@example.com",
            org_id="org-001", is_pending=True,
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        mock_user_store.add_user_to_org.return_value = False
        _patch_user(monkeypatch, MockUser(email="admin@example.com", org_id=None))
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 500

    def test_unauthenticated(self, handler, monkeypatch):
        _patch_user_none(monkeypatch)
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 401

    def test_method_not_allowed(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="GET"
        )
        assert _status(result) == 405

    def test_no_user_store(self, handler_no_store, monkeypatch):
        inv = MockInvitation(
            token="abc-token", email="admin@example.com",
            org_id="org-001", is_pending=True,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_current_user",
            lambda self, handler: (MockUser(email="admin@example.com", org_id=None), MockAuthContext()),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.organizations.OrganizationsHandler._get_invitation_by_token",
            lambda self, token: inv,
        )
        h = MockHTTPHandler(method="POST")
        result = handler_no_store.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 503


# ===========================================================================
# Rate limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting on organization endpoints."""

    def test_rate_limit_exceeded(self, handler, monkeypatch):
        from aragora.server.handlers.organizations import _org_limiter

        monkeypatch.setattr(_org_limiter, "is_allowed", lambda ip: False)
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org/org-001", {}, h, method="GET")
        assert _status(result) == 429


# ===========================================================================
# Role hierarchy
# ===========================================================================


class TestRoleHierarchy:
    """Test role hierarchy constants."""

    def test_member_lowest(self):
        assert ROLE_HIERARCHY["member"] < ROLE_HIERARCHY["admin"]

    def test_admin_middle(self):
        assert ROLE_HIERARCHY["admin"] < ROLE_HIERARCHY["owner"]

    def test_owner_highest(self):
        assert ROLE_HIERARCHY["owner"] == 3


# ===========================================================================
# _check_org_access tests
# ===========================================================================


class TestCheckOrgAccess:
    """Test _check_org_access method."""

    def test_no_user(self, handler):
        ok, err = handler._check_org_access(None, "org-001")
        assert ok is False
        assert "Authentication" in err

    def test_wrong_org(self, handler):
        user = MockUser(org_id="org-999")
        ok, err = handler._check_org_access(user, "org-001")
        assert ok is False
        assert "Not a member" in err

    def test_insufficient_role(self, handler):
        user = MockUser(org_id="org-001", role="member")
        ok, err = handler._check_org_access(user, "org-001", min_role="admin")
        assert ok is False
        assert "admin" in err

    def test_member_access(self, handler):
        user = MockUser(org_id="org-001", role="member")
        ok, err = handler._check_org_access(user, "org-001")
        assert ok is True

    def test_admin_access(self, handler):
        user = MockUser(org_id="org-001", role="admin")
        ok, err = handler._check_org_access(user, "org-001", min_role="admin")
        assert ok is True

    def test_owner_access_for_admin_level(self, handler):
        user = MockUser(org_id="org-001", role="owner")
        ok, err = handler._check_org_access(user, "org-001", min_role="admin")
        assert ok is True

    def test_unknown_role(self, handler):
        user = MockUser(org_id="org-001", role="unknown")
        ok, err = handler._check_org_access(user, "org-001", min_role="admin")
        assert ok is False


# ===========================================================================
# Handler initialization
# ===========================================================================


class TestHandlerInit:
    """Test handler initialization."""

    def test_default_ctx(self):
        h = OrganizationsHandler()
        assert h.ctx == {}

    def test_ctx_passed(self):
        h = OrganizationsHandler(ctx={"key": "value"})
        assert h.ctx["key"] == "value"

    def test_resource_type(self):
        assert OrganizationsHandler.RESOURCE_TYPE == "organization"

    def test_routes_defined(self):
        assert len(OrganizationsHandler.ROUTES) > 0


# ===========================================================================
# Unmatched paths
# ===========================================================================


class TestUnmatchedPaths:
    """Test that unmatched paths return None."""

    def test_unmatched_returns_none(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/v1/debates", {}, h, method="GET")
        assert result is None

    def test_partial_org_path(self, handler):
        h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/org", {}, h, method="GET")
        assert result is None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_handler_uses_command_attribute(self, handler, mock_user_store):
        """Test that handler.command overrides method parameter."""
        h = MockHTTPHandler(method="GET")
        h.command = "GET"
        result = handler.handle("/api/org/org-001", {}, h, method="POST")
        # handler.command is GET, so it should act as GET
        assert _status(result) == 200

    def test_whitespace_org_id_in_switch(self, handler):
        h = MockHTTPHandler(body={"org_id": "   "}, method="POST")
        result = handler.handle(
            "/api/user/organizations/switch", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_whitespace_org_id_in_default(self, handler):
        h = MockHTTPHandler(body={"org_id": "   "}, method="POST")
        result = handler.handle(
            "/api/user/organizations/default", {}, h, method="POST"
        )
        assert _status(result) == 400

    def test_name_only_whitespace(self, handler):
        h = MockHTTPHandler(body={"name": "  X "}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        # "X" after strip is 1 char, must be >= 2
        assert _status(result) == 400

    def test_settings_non_string_value_not_checked_for_size(self, handler, mock_user_store):
        """Non-string values in settings are not checked for size."""
        settings = {"number": 42, "list": [1, 2, 3]}
        h = MockHTTPHandler(body={"settings": settings}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        assert _status(result) == 200

    def test_settings_as_non_dict_ignored(self, handler):
        """settings value that is not a dict should be ignored."""
        h = MockHTTPHandler(body={"settings": "not-a-dict"}, method="PUT")
        result = handler.handle("/api/org/org-001", {}, h, method="PUT")
        # settings is not a dict -> ignored -> no valid fields -> 400
        assert _status(result) == 400

    def test_case_insensitive_email_match_on_accept(self, handler, mock_user_store, monkeypatch):
        inv = MockInvitation(
            token="abc-token", email="Admin@Example.com",
            org_id="org-001", is_pending=True,
        )
        mock_user_store.get_invitation_by_token.return_value = inv
        _patch_user(monkeypatch, MockUser(email="admin@example.com", org_id=None))
        h = MockHTTPHandler(method="POST")
        result = handler.handle(
            "/api/invitations/abc-token/accept", {}, h, method="POST"
        )
        assert _status(result) == 200
