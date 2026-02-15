"""Tests for OrganizationsAPI client resource."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.organizations import (
    Organization,
    OrganizationInvitation,
    OrganizationMember,
    OrganizationsAPI,
    UserOrganizationMembership,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> OrganizationsAPI:
    return OrganizationsAPI(mock_client)


# ---------------------------------------------------------------------------
# Sample response data
# ---------------------------------------------------------------------------

SAMPLE_ORG = {
    "id": "org-abc123",
    "name": "Acme Corp",
    "slug": "acme-corp",
    "tier": "professional",
    "owner_id": "user-owner-1",
    "member_count": 12,
    "debates_used": 45,
    "debates_limit": 100,
    "settings": {"default_agent_count": 5, "allow_public_debates": True},
    "created_at": "2026-01-10T08:00:00Z",
}

SAMPLE_MEMBER = {
    "id": "user-m1",
    "email": "alice@acme.com",
    "name": "Alice Smith",
    "role": "admin",
    "is_active": True,
    "created_at": "2026-01-11T09:00:00Z",
    "last_login_at": "2026-02-10T14:30:00Z",
}

SAMPLE_INVITATION = {
    "id": "inv-001",
    "org_id": "org-abc123",
    "email": "bob@example.com",
    "role": "member",
    "status": "pending",
    "invited_by": "user-owner-1",
    "expires_at": "2026-03-10T08:00:00Z",
    "created_at": "2026-02-10T08:00:00Z",
    "accepted_at": None,
}

SAMPLE_MEMBERSHIP = {
    "user_id": "user-m1",
    "org_id": "org-abc123",
    "organization": SAMPLE_ORG,
    "role": "admin",
    "is_default": True,
    "joined_at": "2026-01-15T12:00:00Z",
}


# ===========================================================================
# Dataclass construction and defaults
# ===========================================================================


class TestDataclasses:
    def test_organization_defaults(self) -> None:
        org = Organization(id="o1", name="Test", slug="test", tier="free", owner_id="u1")
        assert org.member_count == 0
        assert org.debates_used == 0
        assert org.debates_limit == 0
        assert org.settings == {}
        assert org.created_at is None

    def test_organization_full(self) -> None:
        now = datetime.now(tz=timezone.utc)
        org = Organization(
            id="o1",
            name="Full",
            slug="full",
            tier="enterprise",
            owner_id="u1",
            member_count=50,
            debates_used=200,
            debates_limit=500,
            settings={"k": "v"},
            created_at=now,
        )
        assert org.tier == "enterprise"
        assert org.settings == {"k": "v"}
        assert org.created_at == now

    def test_organization_member_defaults(self) -> None:
        member = OrganizationMember(id="m1", email="a@b.com")
        assert member.name is None
        assert member.role == "member"
        assert member.is_active is True
        assert member.created_at is None
        assert member.last_login_at is None

    def test_organization_member_full(self) -> None:
        now = datetime.now(tz=timezone.utc)
        member = OrganizationMember(
            id="m1",
            email="a@b.com",
            name="Alice",
            role="admin",
            is_active=False,
            created_at=now,
            last_login_at=now,
        )
        assert member.name == "Alice"
        assert member.role == "admin"
        assert member.is_active is False

    def test_organization_invitation_required_fields(self) -> None:
        now = datetime.now(tz=timezone.utc)
        inv = OrganizationInvitation(
            id="i1",
            org_id="o1",
            email="x@y.com",
            role="member",
            status="pending",
            invited_by="u1",
            expires_at=now,
        )
        assert inv.created_at is None
        assert inv.accepted_at is None

    def test_user_organization_membership_defaults(self) -> None:
        org = Organization(id="o1", name="T", slug="t", tier="free", owner_id="u1")
        mem = UserOrganizationMembership(user_id="u1", org_id="o1", organization=org, role="member")
        assert mem.is_default is False
        assert mem.joined_at is None


# ===========================================================================
# Organization CRUD
# ===========================================================================


class TestOrganizationGet:
    def test_get(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"organization": SAMPLE_ORG}
        org = api.get("org-abc123")
        assert isinstance(org, Organization)
        assert org.id == "org-abc123"
        assert org.name == "Acme Corp"
        assert org.slug == "acme-corp"
        assert org.tier == "professional"
        assert org.owner_id == "user-owner-1"
        assert org.member_count == 12
        assert org.debates_used == 45
        assert org.debates_limit == 100
        assert org.settings == {"default_agent_count": 5, "allow_public_debates": True}
        assert org.created_at is not None
        assert org.created_at.year == 2026
        mock_client._get.assert_called_once_with("/api/v1/org/org-abc123")

    def test_get_unwrapped_response(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """When the response has no 'organization' key, the whole dict is used."""
        mock_client._get.return_value = SAMPLE_ORG
        org = api.get("org-abc123")
        assert org.id == "org-abc123"

    @pytest.mark.asyncio
    async def test_get_async(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"organization": SAMPLE_ORG})
        org = await api.get_async("org-abc123")
        assert org.id == "org-abc123"
        assert org.name == "Acme Corp"
        mock_client._get_async.assert_called_once_with("/api/v1/org/org-abc123")

    @pytest.mark.asyncio
    async def test_get_async_unwrapped(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_ORG)
        org = await api.get_async("org-abc123")
        assert org.slug == "acme-corp"


class TestOrganizationUpdate:
    def test_update_name(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"organization": {**SAMPLE_ORG, "name": "New Name"}}
        org = api.update("org-abc123", name="New Name")
        assert org.name == "New Name"
        mock_client._put.assert_called_once_with("/api/v1/org/org-abc123", {"name": "New Name"})

    def test_update_settings(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        new_settings = {"max_agents": 10}
        mock_client._put.return_value = {"organization": {**SAMPLE_ORG, "settings": new_settings}}
        org = api.update("org-abc123", settings=new_settings)
        assert org.settings == new_settings
        body = mock_client._put.call_args[0][1]
        assert body == {"settings": new_settings}

    def test_update_both(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"organization": SAMPLE_ORG}
        api.update("org-abc123", name="N", settings={"k": "v"})
        body = mock_client._put.call_args[0][1]
        assert body == {"name": "N", "settings": {"k": "v"}}

    def test_update_nothing(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        """Passing no optional args sends an empty body."""
        mock_client._put.return_value = {"organization": SAMPLE_ORG}
        api.update("org-abc123")
        body = mock_client._put.call_args[0][1]
        assert body == {}

    @pytest.mark.asyncio
    async def test_update_async(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._put_async = AsyncMock(
            return_value={"organization": {**SAMPLE_ORG, "name": "Async Name"}}
        )
        org = await api.update_async("org-abc123", name="Async Name")
        assert org.name == "Async Name"
        mock_client._put_async.assert_called_once_with(
            "/api/v1/org/org-abc123", {"name": "Async Name"}
        )

    @pytest.mark.asyncio
    async def test_update_async_empty_body(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._put_async = AsyncMock(return_value={"organization": SAMPLE_ORG})
        await api.update_async("org-abc123")
        body = mock_client._put_async.call_args[0][1]
        assert body == {}


# ===========================================================================
# Member Management
# ===========================================================================


class TestListMembers:
    def test_list_members(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"members": [SAMPLE_MEMBER]}
        members = api.list_members("org-abc123")
        assert len(members) == 1
        assert isinstance(members[0], OrganizationMember)
        assert members[0].id == "user-m1"
        assert members[0].email == "alice@acme.com"
        assert members[0].name == "Alice Smith"
        assert members[0].role == "admin"
        mock_client._get.assert_called_once_with("/api/v1/org/org-abc123/members")

    def test_list_members_empty(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"members": []}
        members = api.list_members("org-abc123")
        assert members == []

    def test_list_members_missing_key(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """When 'members' key is missing, should return empty list."""
        mock_client._get.return_value = {}
        members = api.list_members("org-abc123")
        assert members == []

    def test_list_members_multiple(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        second_member = {
            "id": "user-m2",
            "email": "carol@acme.com",
            "role": "member",
        }
        mock_client._get.return_value = {"members": [SAMPLE_MEMBER, second_member]}
        members = api.list_members("org-abc123")
        assert len(members) == 2
        assert members[1].email == "carol@acme.com"
        assert members[1].role == "member"
        assert members[1].name is None  # not provided => default from .get("name")

    @pytest.mark.asyncio
    async def test_list_members_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"members": [SAMPLE_MEMBER]})
        members = await api.list_members_async("org-abc123")
        assert len(members) == 1
        assert members[0].email == "alice@acme.com"


class TestInviteMember:
    def test_invite_default_role(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"invitation_id": "inv-new", "expires_in": 72}
        result = api.invite_member("org-abc123", "bob@example.com")
        assert result == {"invitation_id": "inv-new", "expires_in": 72}
        mock_client._post.assert_called_once_with(
            "/api/v1/org/org-abc123/invite",
            {"email": "bob@example.com", "role": "member"},
        )

    def test_invite_admin_role(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"invitation_id": "inv-adm"}
        api.invite_member("org-abc123", "admin@example.com", role="admin")
        body = mock_client._post.call_args[0][1]
        assert body["role"] == "admin"

    @pytest.mark.asyncio
    async def test_invite_member_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"invitation_id": "inv-async"})
        result = await api.invite_member_async("org-abc123", "new@example.com")
        assert result["invitation_id"] == "inv-async"
        mock_client._post_async.assert_called_once_with(
            "/api/v1/org/org-abc123/invite",
            {"email": "new@example.com", "role": "member"},
        )


class TestRemoveMember:
    def test_remove_member(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = {}
        result = api.remove_member("org-abc123", "user-m1")
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/org/org-abc123/members/user-m1")

    @pytest.mark.asyncio
    async def test_remove_member_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._delete_async = AsyncMock(return_value={})
        result = await api.remove_member_async("org-abc123", "user-m1")
        assert result is True
        mock_client._delete_async.assert_called_once_with("/api/v1/org/org-abc123/members/user-m1")


class TestUpdateMemberRole:
    def test_update_member_role(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._put.return_value = {"user_id": "user-m1", "role": "admin"}
        result = api.update_member_role("org-abc123", "user-m1", "admin")
        assert result["role"] == "admin"
        mock_client._put.assert_called_once_with(
            "/api/v1/org/org-abc123/members/user-m1/role", {"role": "admin"}
        )

    @pytest.mark.asyncio
    async def test_update_member_role_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._put_async = AsyncMock(return_value={"user_id": "user-m1", "role": "member"})
        result = await api.update_member_role_async("org-abc123", "user-m1", "member")
        assert result["role"] == "member"


# ===========================================================================
# Invitation Management
# ===========================================================================


class TestListInvitations:
    def test_list_invitations(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"invitations": [SAMPLE_INVITATION]}
        invitations = api.list_invitations("org-abc123")
        assert len(invitations) == 1
        inv = invitations[0]
        assert isinstance(inv, OrganizationInvitation)
        assert inv.id == "inv-001"
        assert inv.org_id == "org-abc123"
        assert inv.email == "bob@example.com"
        assert inv.role == "member"
        assert inv.status == "pending"
        assert inv.invited_by == "user-owner-1"
        assert inv.expires_at.year == 2026
        assert inv.created_at is not None
        mock_client._get.assert_called_once_with("/api/v1/org/org-abc123/invitations")

    def test_list_invitations_empty(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"invitations": []}
        invitations = api.list_invitations("org-abc123")
        assert invitations == []

    def test_list_invitations_missing_key(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        invitations = api.list_invitations("org-abc123")
        assert invitations == []

    @pytest.mark.asyncio
    async def test_list_invitations_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"invitations": [SAMPLE_INVITATION]})
        invitations = await api.list_invitations_async("org-abc123")
        assert len(invitations) == 1
        assert invitations[0].email == "bob@example.com"


class TestRevokeInvitation:
    def test_revoke_invitation(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = {}
        result = api.revoke_invitation("org-abc123", "inv-001")
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/org/org-abc123/invitations/inv-001")

    @pytest.mark.asyncio
    async def test_revoke_invitation_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._delete_async = AsyncMock(return_value={})
        result = await api.revoke_invitation_async("org-abc123", "inv-001")
        assert result is True
        mock_client._delete_async.assert_called_once_with(
            "/api/v1/org/org-abc123/invitations/inv-001"
        )


class TestGetPendingInvitations:
    def test_get_pending_invitations(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"invitations": [SAMPLE_INVITATION]}
        invitations = api.get_pending_invitations()
        assert len(invitations) == 1
        assert invitations[0].status == "pending"
        mock_client._get.assert_called_once_with("/api/v1/invitations/pending")

    def test_get_pending_invitations_empty(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        invitations = api.get_pending_invitations()
        assert invitations == []

    @pytest.mark.asyncio
    async def test_get_pending_invitations_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"invitations": [SAMPLE_INVITATION]})
        invitations = await api.get_pending_invitations_async()
        assert len(invitations) == 1


class TestAcceptInvitation:
    def test_accept_invitation(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {
            "organization": SAMPLE_ORG,
            "role": "member",
        }
        result = api.accept_invitation("tok-abc")
        assert result["role"] == "member"
        mock_client._post.assert_called_once_with("/api/v1/invitations/tok-abc/accept", {})

    @pytest.mark.asyncio
    async def test_accept_invitation_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(
            return_value={"organization": SAMPLE_ORG, "role": "member"}
        )
        result = await api.accept_invitation_async("tok-abc")
        assert result["role"] == "member"
        mock_client._post_async.assert_called_once_with("/api/v1/invitations/tok-abc/accept", {})


# ===========================================================================
# User Organization Management
# ===========================================================================


class TestListUserOrganizations:
    def test_list_user_organizations(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"organizations": [SAMPLE_MEMBERSHIP]}
        memberships = api.list_user_organizations()
        assert len(memberships) == 1
        m = memberships[0]
        assert isinstance(m, UserOrganizationMembership)
        assert m.user_id == "user-m1"
        assert m.org_id == "org-abc123"
        assert m.role == "admin"
        assert m.is_default is True
        assert isinstance(m.organization, Organization)
        assert m.organization.name == "Acme Corp"
        assert m.joined_at is not None
        mock_client._get.assert_called_once_with("/api/v1/user/organizations")

    def test_list_user_organizations_empty(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"organizations": []}
        memberships = api.list_user_organizations()
        assert memberships == []

    def test_list_user_organizations_missing_key(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {}
        memberships = api.list_user_organizations()
        assert memberships == []

    @pytest.mark.asyncio
    async def test_list_user_organizations_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(return_value={"organizations": [SAMPLE_MEMBERSHIP]})
        memberships = await api.list_user_organizations_async()
        assert len(memberships) == 1
        assert memberships[0].organization.slug == "acme-corp"


class TestSwitchOrganization:
    def test_switch_organization(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"organization": SAMPLE_ORG}
        org = api.switch_organization("org-abc123")
        assert isinstance(org, Organization)
        assert org.id == "org-abc123"
        mock_client._post.assert_called_once_with(
            "/api/v1/user/organizations/switch", {"org_id": "org-abc123"}
        )

    def test_switch_organization_unwrapped(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = SAMPLE_ORG
        org = api.switch_organization("org-abc123")
        assert org.name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_switch_organization_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"organization": SAMPLE_ORG})
        org = await api.switch_organization_async("org-abc123")
        assert org.id == "org-abc123"


class TestSetDefaultOrganization:
    def test_set_default_organization(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"success": True}
        result = api.set_default_organization("org-abc123")
        assert result is True
        mock_client._post.assert_called_once_with(
            "/api/v1/user/organizations/default", {"org_id": "org-abc123"}
        )

    @pytest.mark.asyncio
    async def test_set_default_organization_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"success": True})
        result = await api.set_default_organization_async("org-abc123")
        assert result is True
        mock_client._post_async.assert_called_once_with(
            "/api/v1/user/organizations/default", {"org_id": "org-abc123"}
        )


class TestLeaveOrganization:
    def test_leave_organization(self, api: OrganizationsAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = {}
        result = api.leave_organization("org-abc123")
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/user/organizations/org-abc123")

    @pytest.mark.asyncio
    async def test_leave_organization_async(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._delete_async = AsyncMock(return_value={})
        result = await api.leave_organization_async("org-abc123")
        assert result is True
        mock_client._delete_async.assert_called_once_with("/api/v1/user/organizations/org-abc123")


# ===========================================================================
# Parsing helpers - edge cases
# ===========================================================================


class TestParseOrganization:
    def test_parse_full_data(self, api: OrganizationsAPI) -> None:
        org = api._parse_organization(SAMPLE_ORG)
        assert org.id == "org-abc123"
        assert org.created_at is not None
        assert org.created_at.year == 2026
        assert org.created_at.month == 1

    def test_parse_minimal_data(self, api: OrganizationsAPI) -> None:
        org = api._parse_organization({})
        assert org.id == ""
        assert org.name == ""
        assert org.slug == ""
        assert org.tier == "free"
        assert org.owner_id == ""
        assert org.member_count == 0
        assert org.created_at is None

    def test_parse_invalid_datetime(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_ORG, "created_at": "not-a-date"}
        org = api._parse_organization(data)
        assert org.created_at is None

    def test_parse_datetime_with_timezone(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_ORG, "created_at": "2026-06-15T10:30:00+05:00"}
        org = api._parse_organization(data)
        assert org.created_at is not None
        assert org.created_at.hour == 10

    def test_parse_none_created_at(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_ORG, "created_at": None}
        org = api._parse_organization(data)
        assert org.created_at is None

    def test_parse_empty_string_created_at(self, api: OrganizationsAPI) -> None:
        """Empty string is falsy, so created_at should remain None."""
        data = {**SAMPLE_ORG, "created_at": ""}
        org = api._parse_organization(data)
        assert org.created_at is None


class TestParseMember:
    def test_parse_full_member(self, api: OrganizationsAPI) -> None:
        member = api._parse_member(SAMPLE_MEMBER)
        assert member.id == "user-m1"
        assert member.name == "Alice Smith"
        assert member.last_login_at is not None
        assert member.last_login_at.month == 2

    def test_parse_minimal_member(self, api: OrganizationsAPI) -> None:
        member = api._parse_member({})
        assert member.id == ""
        assert member.email == ""
        assert member.name is None
        assert member.role == "member"
        assert member.is_active is True
        assert member.created_at is None
        assert member.last_login_at is None

    def test_parse_member_invalid_created_at(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_MEMBER, "created_at": "bad-date"}
        member = api._parse_member(data)
        assert member.created_at is None

    def test_parse_member_invalid_last_login(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_MEMBER, "last_login_at": "bad-date"}
        member = api._parse_member(data)
        assert member.last_login_at is None

    def test_parse_member_inactive(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_MEMBER, "is_active": False}
        member = api._parse_member(data)
        assert member.is_active is False


class TestParseInvitation:
    def test_parse_full_invitation(self, api: OrganizationsAPI) -> None:
        inv = api._parse_invitation(SAMPLE_INVITATION)
        assert inv.id == "inv-001"
        assert inv.org_id == "org-abc123"
        assert inv.email == "bob@example.com"
        assert inv.status == "pending"
        assert inv.expires_at.year == 2026
        assert inv.created_at is not None
        assert inv.accepted_at is None  # None in sample data

    def test_parse_minimal_invitation(self, api: OrganizationsAPI) -> None:
        inv = api._parse_invitation({})
        assert inv.id == ""
        assert inv.org_id == ""
        assert inv.email == ""
        assert inv.role == "member"
        assert inv.status == "pending"
        assert inv.invited_by == ""
        # expires_at defaults to datetime.now() when not provided
        assert isinstance(inv.expires_at, datetime)
        assert inv.created_at is None
        assert inv.accepted_at is None

    def test_parse_invitation_with_accepted(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_INVITATION, "accepted_at": "2026-02-11T10:00:00Z"}
        inv = api._parse_invitation(data)
        assert inv.accepted_at is not None
        assert inv.accepted_at.day == 11

    def test_parse_invitation_invalid_expires(self, api: OrganizationsAPI) -> None:
        """Invalid expires_at still produces a datetime (the default now())."""
        data = {**SAMPLE_INVITATION, "expires_at": "garbage"}
        inv = api._parse_invitation(data)
        # Falls back to datetime.now() set at the top of _parse_invitation
        assert isinstance(inv.expires_at, datetime)

    def test_parse_invitation_invalid_accepted_at(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_INVITATION, "accepted_at": "garbage"}
        inv = api._parse_invitation(data)
        assert inv.accepted_at is None


class TestParseMembership:
    def test_parse_full_membership(self, api: OrganizationsAPI) -> None:
        mem = api._parse_membership(SAMPLE_MEMBERSHIP)
        assert mem.user_id == "user-m1"
        assert mem.org_id == "org-abc123"
        assert mem.role == "admin"
        assert mem.is_default is True
        assert mem.joined_at is not None
        assert isinstance(mem.organization, Organization)
        assert mem.organization.name == "Acme Corp"

    def test_parse_minimal_membership(self, api: OrganizationsAPI) -> None:
        mem = api._parse_membership({})
        assert mem.user_id == ""
        assert mem.org_id == ""
        assert mem.role == "member"
        assert mem.is_default is False
        assert mem.joined_at is None
        # The embedded organization should be parsed from an empty dict
        assert isinstance(mem.organization, Organization)
        assert mem.organization.id == ""
        assert mem.organization.tier == "free"

    def test_parse_membership_invalid_joined_at(self, api: OrganizationsAPI) -> None:
        data = {**SAMPLE_MEMBERSHIP, "joined_at": "not-valid"}
        mem = api._parse_membership(data)
        assert mem.joined_at is None

    def test_parse_membership_nested_org(self, api: OrganizationsAPI) -> None:
        """The nested organization is fully parsed."""
        data = {
            "user_id": "u1",
            "org_id": "o1",
            "role": "member",
            "organization": {
                "id": "o1",
                "name": "Inner Org",
                "slug": "inner-org",
                "tier": "enterprise",
                "owner_id": "owner1",
                "member_count": 99,
                "created_at": "2025-12-01T00:00:00Z",
            },
        }
        mem = api._parse_membership(data)
        assert mem.organization.name == "Inner Org"
        assert mem.organization.tier == "enterprise"
        assert mem.organization.member_count == 99
        assert mem.organization.created_at is not None


# ===========================================================================
# Integration-like workflow tests
# ===========================================================================


class TestOrganizationWorkflows:
    def test_create_org_invite_members_workflow(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """Simulate: get org -> invite members -> list members."""
        # Step 1: Get organization
        mock_client._get.return_value = {"organization": SAMPLE_ORG}
        org = api.get("org-abc123")
        assert org.name == "Acme Corp"

        # Step 2: Invite two members
        mock_client._post.return_value = {"invitation_id": "inv-new-1"}
        result1 = api.invite_member("org-abc123", "charlie@example.com")
        assert result1["invitation_id"] == "inv-new-1"

        mock_client._post.return_value = {"invitation_id": "inv-new-2"}
        result2 = api.invite_member("org-abc123", "diana@example.com", role="admin")
        assert result2["invitation_id"] == "inv-new-2"

        # Step 3: List members
        mock_client._get.return_value = {
            "members": [
                SAMPLE_MEMBER,
                {"id": "user-m3", "email": "charlie@example.com", "role": "member"},
            ]
        }
        members = api.list_members("org-abc123")
        assert len(members) == 2

    def test_switch_org_and_set_default_workflow(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """Simulate: list user orgs -> switch -> set default."""
        # Step 1: List user orgs
        second_org = {
            **SAMPLE_ORG,
            "id": "org-beta",
            "name": "Beta Inc",
            "slug": "beta-inc",
        }
        second_membership = {
            "user_id": "user-m1",
            "org_id": "org-beta",
            "organization": second_org,
            "role": "member",
            "is_default": False,
        }
        mock_client._get.return_value = {"organizations": [SAMPLE_MEMBERSHIP, second_membership]}
        memberships = api.list_user_organizations()
        assert len(memberships) == 2
        target_org_id = memberships[1].org_id

        # Step 2: Switch to beta org
        mock_client._post.return_value = {"organization": second_org}
        org = api.switch_organization(target_org_id)
        assert org.name == "Beta Inc"

        # Step 3: Set as default
        mock_client._post.return_value = {"success": True}
        result = api.set_default_organization(target_org_id)
        assert result is True

    def test_accept_invitation_and_check_membership(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """Simulate: get pending invitations -> accept -> verify membership."""
        # Step 1: Get pending invitations
        mock_client._get.return_value = {"invitations": [SAMPLE_INVITATION]}
        pending = api.get_pending_invitations()
        assert len(pending) == 1

        # Step 2: Accept the invitation
        mock_client._post.return_value = {
            "organization": SAMPLE_ORG,
            "role": "member",
        }
        accept_result = api.accept_invitation("tok-123")
        assert accept_result["role"] == "member"

        # Step 3: Verify membership
        mock_client._get.return_value = {"organizations": [SAMPLE_MEMBERSHIP]}
        memberships = api.list_user_organizations()
        assert len(memberships) == 1
        assert memberships[0].organization.id == "org-abc123"

    def test_remove_member_and_revoke_invitation_workflow(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """Simulate: remove member + revoke their pending invitation."""
        # Remove the member
        mock_client._delete.return_value = {}
        assert api.remove_member("org-abc123", "user-m1") is True

        # Revoke any pending invitation for that user
        assert api.revoke_invitation("org-abc123", "inv-001") is True
        assert mock_client._delete.call_count == 2

    def test_update_role_then_leave_workflow(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """Simulate: promote member to admin -> member leaves."""
        # Promote
        mock_client._put.return_value = {"user_id": "user-m2", "role": "admin"}
        result = api.update_member_role("org-abc123", "user-m2", "admin")
        assert result["role"] == "admin"

        # Member leaves
        mock_client._delete.return_value = {}
        assert api.leave_organization("org-abc123") is True

    @pytest.mark.asyncio
    async def test_async_full_workflow(
        self, api: OrganizationsAPI, mock_client: AragoraClient
    ) -> None:
        """Full async workflow: get org -> list members -> invite -> update role."""
        # Get org
        mock_client._get_async = AsyncMock(return_value={"organization": SAMPLE_ORG})
        org = await api.get_async("org-abc123")
        assert org.name == "Acme Corp"

        # List members
        mock_client._get_async = AsyncMock(return_value={"members": [SAMPLE_MEMBER]})
        members = await api.list_members_async("org-abc123")
        assert len(members) == 1

        # Invite
        mock_client._post_async = AsyncMock(return_value={"invitation_id": "inv-async-1"})
        inv_result = await api.invite_member_async("org-abc123", "new@test.com")
        assert inv_result["invitation_id"] == "inv-async-1"

        # Update role
        mock_client._put_async = AsyncMock(return_value={"user_id": "user-m1", "role": "admin"})
        role_result = await api.update_member_role_async("org-abc123", "user-m1", "admin")
        assert role_result["role"] == "admin"


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    def test_all_exports(self) -> None:
        from aragora.client.resources import organizations

        assert "OrganizationsAPI" in organizations.__all__
        assert "Organization" in organizations.__all__
        assert "OrganizationMember" in organizations.__all__
        assert "OrganizationInvitation" in organizations.__all__
        assert "UserOrganizationMembership" in organizations.__all__
        assert len(organizations.__all__) == 5
