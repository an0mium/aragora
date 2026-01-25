"""
Organizations API resource for the Aragora client.

Provides methods for organization management:
- Get/update organization details
- Member management (list, invite, remove, update roles)
- Invitation management
- User organization switching
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class Organization:
    """An organization."""

    id: str
    name: str
    slug: str
    tier: str
    owner_id: str
    member_count: int = 0
    debates_used: int = 0
    debates_limit: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class OrganizationMember:
    """A member of an organization."""

    id: str
    email: str
    name: Optional[str] = None
    role: str = "member"
    is_active: bool = True
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


@dataclass
class OrganizationInvitation:
    """An invitation to join an organization."""

    id: str
    org_id: str
    email: str
    role: str
    status: str
    invited_by: str
    expires_at: datetime
    created_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None


@dataclass
class UserOrganizationMembership:
    """A user's membership in an organization."""

    user_id: str
    org_id: str
    organization: Organization
    role: str
    is_default: bool = False
    joined_at: Optional[datetime] = None


class OrganizationsAPI:
    """API interface for organization management."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Organization CRUD
    # =========================================================================

    def get(self, org_id: str) -> Organization:
        """
        Get organization details.

        Args:
            org_id: The organization ID.

        Returns:
            Organization object.
        """
        response = self._client._get(f"/api/v1/org/{org_id}")
        org_data = response.get("organization", response)
        return self._parse_organization(org_data)

    async def get_async(self, org_id: str) -> Organization:
        """Async version of get()."""
        response = await self._client._get_async(f"/api/v1/org/{org_id}")
        org_data = response.get("organization", response)
        return self._parse_organization(org_data)

    def update(
        self,
        org_id: str,
        name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """
        Update organization settings.

        Args:
            org_id: The organization ID.
            name: New organization name.
            settings: New organization settings.

        Returns:
            Updated Organization object.
        """
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if settings is not None:
            body["settings"] = settings

        response = self._client._put(f"/api/v1/org/{org_id}", body)
        org_data = response.get("organization", response)
        return self._parse_organization(org_data)

    async def update_async(
        self,
        org_id: str,
        name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Organization:
        """Async version of update()."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if settings is not None:
            body["settings"] = settings

        response = await self._client._put_async(f"/api/v1/org/{org_id}", body)
        org_data = response.get("organization", response)
        return self._parse_organization(org_data)

    # =========================================================================
    # Member Management
    # =========================================================================

    def list_members(self, org_id: str) -> List[OrganizationMember]:
        """
        List members of an organization.

        Args:
            org_id: The organization ID.

        Returns:
            List of OrganizationMember objects.
        """
        response = self._client._get(f"/api/v1/org/{org_id}/members")
        members = response.get("members", [])
        return [self._parse_member(m) for m in members]

    async def list_members_async(self, org_id: str) -> List[OrganizationMember]:
        """Async version of list_members()."""
        response = await self._client._get_async(f"/api/v1/org/{org_id}/members")
        members = response.get("members", [])
        return [self._parse_member(m) for m in members]

    def invite_member(
        self,
        org_id: str,
        email: str,
        role: str = "member",
    ) -> Dict[str, Any]:
        """
        Invite a user to the organization.

        Args:
            org_id: The organization ID.
            email: Email address to invite.
            role: Role to assign (member, admin).

        Returns:
            Invitation details including ID and expiry.
        """
        body = {"email": email, "role": role}
        return self._client._post(f"/api/v1/org/{org_id}/invite", body)

    async def invite_member_async(
        self,
        org_id: str,
        email: str,
        role: str = "member",
    ) -> Dict[str, Any]:
        """Async version of invite_member()."""
        body = {"email": email, "role": role}
        return await self._client._post_async(f"/api/v1/org/{org_id}/invite", body)

    def remove_member(self, org_id: str, user_id: str) -> bool:
        """
        Remove a member from the organization.

        Args:
            org_id: The organization ID.
            user_id: The user ID to remove.

        Returns:
            True if successful.
        """
        self._client._delete(f"/api/v1/org/{org_id}/members/{user_id}")
        return True

    async def remove_member_async(self, org_id: str, user_id: str) -> bool:
        """Async version of remove_member()."""
        await self._client._delete_async(f"/api/v1/org/{org_id}/members/{user_id}")
        return True

    def update_member_role(self, org_id: str, user_id: str, role: str) -> Dict[str, Any]:
        """
        Update a member's role in the organization.

        Args:
            org_id: The organization ID.
            user_id: The user ID.
            role: New role (member, admin).

        Returns:
            Updated member details.
        """
        body = {"role": role}
        return self._client._put(f"/api/v1/org/{org_id}/members/{user_id}/role", body)

    async def update_member_role_async(
        self, org_id: str, user_id: str, role: str
    ) -> Dict[str, Any]:
        """Async version of update_member_role()."""
        body = {"role": role}
        return await self._client._put_async(f"/api/v1/org/{org_id}/members/{user_id}/role", body)

    # =========================================================================
    # Invitation Management
    # =========================================================================

    def list_invitations(self, org_id: str) -> List[OrganizationInvitation]:
        """
        List pending invitations for an organization.

        Args:
            org_id: The organization ID.

        Returns:
            List of OrganizationInvitation objects.
        """
        response = self._client._get(f"/api/v1/org/{org_id}/invitations")
        invitations = response.get("invitations", [])
        return [self._parse_invitation(inv) for inv in invitations]

    async def list_invitations_async(self, org_id: str) -> List[OrganizationInvitation]:
        """Async version of list_invitations()."""
        response = await self._client._get_async(f"/api/v1/org/{org_id}/invitations")
        invitations = response.get("invitations", [])
        return [self._parse_invitation(inv) for inv in invitations]

    def revoke_invitation(self, org_id: str, invitation_id: str) -> bool:
        """
        Revoke a pending invitation.

        Args:
            org_id: The organization ID.
            invitation_id: The invitation ID.

        Returns:
            True if successful.
        """
        self._client._delete(f"/api/v1/org/{org_id}/invitations/{invitation_id}")
        return True

    async def revoke_invitation_async(self, org_id: str, invitation_id: str) -> bool:
        """Async version of revoke_invitation()."""
        await self._client._delete_async(f"/api/v1/org/{org_id}/invitations/{invitation_id}")
        return True

    def get_pending_invitations(self) -> List[OrganizationInvitation]:
        """
        Get pending invitations for the current user.

        Returns:
            List of OrganizationInvitation objects.
        """
        response = self._client._get("/api/v1/invitations/pending")
        invitations = response.get("invitations", [])
        return [self._parse_invitation(inv) for inv in invitations]

    async def get_pending_invitations_async(self) -> List[OrganizationInvitation]:
        """Async version of get_pending_invitations()."""
        response = await self._client._get_async("/api/v1/invitations/pending")
        invitations = response.get("invitations", [])
        return [self._parse_invitation(inv) for inv in invitations]

    def accept_invitation(self, token: str) -> Dict[str, Any]:
        """
        Accept an organization invitation.

        Args:
            token: The invitation token.

        Returns:
            Result including organization details and role.
        """
        return self._client._post(f"/api/v1/invitations/{token}/accept", {})

    async def accept_invitation_async(self, token: str) -> Dict[str, Any]:
        """Async version of accept_invitation()."""
        return await self._client._post_async(f"/api/v1/invitations/{token}/accept", {})

    # =========================================================================
    # User Organization Management
    # =========================================================================

    def list_user_organizations(self) -> List[UserOrganizationMembership]:
        """
        List organizations for the current user.

        Returns:
            List of UserOrganizationMembership objects.
        """
        response = self._client._get("/api/v1/user/organizations")
        orgs = response.get("organizations", [])
        return [self._parse_membership(o) for o in orgs]

    async def list_user_organizations_async(self) -> List[UserOrganizationMembership]:
        """Async version of list_user_organizations()."""
        response = await self._client._get_async("/api/v1/user/organizations")
        orgs = response.get("organizations", [])
        return [self._parse_membership(o) for o in orgs]

    def switch_organization(self, org_id: str) -> Organization:
        """
        Switch the active organization for the current user.

        Args:
            org_id: The organization ID to switch to.

        Returns:
            The new active Organization.
        """
        body = {"org_id": org_id}
        response = self._client._post("/api/v1/user/organizations/switch", body)
        org_data = response.get("organization", response)
        return self._parse_organization(org_data)

    async def switch_organization_async(self, org_id: str) -> Organization:
        """Async version of switch_organization()."""
        body = {"org_id": org_id}
        response = await self._client._post_async("/api/v1/user/organizations/switch", body)
        org_data = response.get("organization", response)
        return self._parse_organization(org_data)

    def set_default_organization(self, org_id: str) -> bool:
        """
        Set the default organization for the current user.

        Args:
            org_id: The organization ID.

        Returns:
            True if successful.
        """
        body = {"org_id": org_id}
        self._client._post("/api/v1/user/organizations/default", body)
        return True

    async def set_default_organization_async(self, org_id: str) -> bool:
        """Async version of set_default_organization()."""
        body = {"org_id": org_id}
        await self._client._post_async("/api/v1/user/organizations/default", body)
        return True

    def leave_organization(self, org_id: str) -> bool:
        """
        Leave an organization.

        Args:
            org_id: The organization ID to leave.

        Returns:
            True if successful.
        """
        self._client._delete(f"/api/v1/user/organizations/{org_id}")
        return True

    async def leave_organization_async(self, org_id: str) -> bool:
        """Async version of leave_organization()."""
        await self._client._delete_async(f"/api/v1/user/organizations/{org_id}")
        return True

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_organization(self, data: Dict[str, Any]) -> Organization:
        """Parse organization data into Organization object."""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Organization(
            id=data.get("id", ""),
            name=data.get("name", ""),
            slug=data.get("slug", ""),
            tier=data.get("tier", "free"),
            owner_id=data.get("owner_id", ""),
            member_count=data.get("member_count", 0),
            debates_used=data.get("debates_used", 0),
            debates_limit=data.get("debates_limit", 0),
            settings=data.get("settings", {}),
            created_at=created_at,
        )

    def _parse_member(self, data: Dict[str, Any]) -> OrganizationMember:
        """Parse member data into OrganizationMember object."""
        created_at = None
        last_login_at = None

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("last_login_at"):
            try:
                last_login_at = datetime.fromisoformat(data["last_login_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return OrganizationMember(
            id=data.get("id", ""),
            email=data.get("email", ""),
            name=data.get("name"),
            role=data.get("role", "member"),
            is_active=data.get("is_active", True),
            created_at=created_at,
            last_login_at=last_login_at,
        )

    def _parse_invitation(self, data: Dict[str, Any]) -> OrganizationInvitation:
        """Parse invitation data into OrganizationInvitation object."""
        expires_at = datetime.now()
        created_at = None
        accepted_at = None

        if data.get("expires_at"):
            try:
                expires_at = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("accepted_at"):
            try:
                accepted_at = datetime.fromisoformat(data["accepted_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return OrganizationInvitation(
            id=data.get("id", ""),
            org_id=data.get("org_id", ""),
            email=data.get("email", ""),
            role=data.get("role", "member"),
            status=data.get("status", "pending"),
            invited_by=data.get("invited_by", ""),
            expires_at=expires_at,
            created_at=created_at,
            accepted_at=accepted_at,
        )

    def _parse_membership(self, data: Dict[str, Any]) -> UserOrganizationMembership:
        """Parse membership data into UserOrganizationMembership object."""
        joined_at = None
        if data.get("joined_at"):
            try:
                joined_at = datetime.fromisoformat(data["joined_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        org_data = data.get("organization", {})

        return UserOrganizationMembership(
            user_id=data.get("user_id", ""),
            org_id=data.get("org_id", ""),
            organization=self._parse_organization(org_data),
            role=data.get("role", "member"),
            is_default=data.get("is_default", False),
            joined_at=joined_at,
        )


__all__ = [
    "OrganizationsAPI",
    "Organization",
    "OrganizationMember",
    "OrganizationInvitation",
    "UserOrganizationMembership",
]
