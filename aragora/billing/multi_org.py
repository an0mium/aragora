"""
Multi-Organization Support.

Enables users to belong to multiple organizations with different roles.
Manages memberships, context switching, and cross-org permissions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from .models import Organization

logger = logging.getLogger(__name__)


class MembershipRole(str, Enum):
    """Roles within an organization."""

    OWNER = "owner"  # Full control, can delete org
    ADMIN = "admin"  # Can manage members, settings
    MEMBER = "member"  # Standard access
    VIEWER = "viewer"  # Read-only access
    BILLING = "billing"  # Can manage billing only


class MembershipStatus(str, Enum):
    """Status of an organization membership."""

    ACTIVE = "active"
    PENDING = "pending"  # Invited but not accepted
    SUSPENDED = "suspended"
    EXPIRED = "expired"


# Role permissions mapping
ROLE_PERMISSIONS: Dict[MembershipRole, Set[str]] = {
    MembershipRole.OWNER: {
        "org.delete",
        "org.update",
        "org.settings",
        "org.billing",
        "members.invite",
        "members.remove",
        "members.promote",
        "members.demote",
        "debates.create",
        "debates.view",
        "debates.delete",
        "api.access",
        "audit.view",
    },
    MembershipRole.ADMIN: {
        "org.update",
        "org.settings",
        "members.invite",
        "members.remove",
        "debates.create",
        "debates.view",
        "debates.delete",
        "api.access",
        "audit.view",
    },
    MembershipRole.MEMBER: {
        "debates.create",
        "debates.view",
        "api.access",
    },
    MembershipRole.VIEWER: {
        "debates.view",
    },
    MembershipRole.BILLING: {
        "org.billing",
        "debates.view",
    },
}


@dataclass
class OrganizationMembership:
    """
    Represents a user's membership in an organization.

    Users can belong to multiple organizations with different roles.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    org_id: str = ""
    role: MembershipRole = MembershipRole.MEMBER
    status: MembershipStatus = MembershipStatus.ACTIVE
    is_primary: bool = False  # User's primary organization
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    invited_by: Optional[str] = None
    invitation_token: Optional[str] = None  # For pending invitations

    # Custom permissions (additions to role)
    custom_permissions: List[str] = field(default_factory=list)

    # Metadata
    display_name: Optional[str] = None  # Name to display in this org
    department: Optional[str] = None
    title: Optional[str] = None

    @property
    def permissions(self) -> Set[str]:
        """Get all permissions for this membership."""
        base_permissions = ROLE_PERMISSIONS.get(self.role, set())
        return base_permissions | set(self.custom_permissions)

    def has_permission(self, permission: str) -> bool:
        """Check if membership has a specific permission."""
        if self.status != MembershipStatus.ACTIVE:
            return False
        return permission in self.permissions

    def can_manage_members(self) -> bool:
        """Check if membership can manage other members."""
        return "members.invite" in self.permissions

    def can_manage_billing(self) -> bool:
        """Check if membership can manage billing."""
        return "org.billing" in self.permissions

    def is_admin_or_owner(self) -> bool:
        """Check if membership has admin or owner role."""
        return self.role in (MembershipRole.OWNER, MembershipRole.ADMIN)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "org_id": self.org_id,
            "role": self.role.value,
            "status": self.status.value,
            "is_primary": self.is_primary,
            "joined_at": self.joined_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "invited_by": self.invited_by,
            "permissions": list(self.permissions),
            "display_name": self.display_name,
            "department": self.department,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrganizationMembership":
        """Create from dictionary."""
        membership = cls(
            id=data.get("id", str(uuid4())),
            user_id=data.get("user_id", ""),
            org_id=data.get("org_id", ""),
            role=MembershipRole(data.get("role", "member")),
            status=MembershipStatus(data.get("status", "active")),
            is_primary=data.get("is_primary", False),
            invited_by=data.get("invited_by"),
            invitation_token=data.get("invitation_token"),
            custom_permissions=data.get("custom_permissions", []),
            display_name=data.get("display_name"),
            department=data.get("department"),
            title=data.get("title"),
        )

        for field_name in ["joined_at", "updated_at"]:
            if field_name in data and data[field_name]:
                value = data[field_name]
                if isinstance(value, str):
                    value = datetime.fromisoformat(value)
                setattr(membership, field_name, value)

        return membership


@dataclass
class OrgContext:
    """Current organization context for a user session."""

    user_id: str
    org_id: str
    membership: OrganizationMembership
    organization: Organization
    switch_count: int = 0  # Number of org switches in session


class MultiOrgManager:
    """
    Manages multi-organization memberships and context switching.

    Features:
    - Users can belong to multiple organizations
    - Each membership has a role with specific permissions
    - One organization is marked as "primary" for the user
    - Context switching between organizations
    - Cross-org permission checks

    Example:
        manager = MultiOrgManager()

        # Add user to organization
        membership = await manager.add_member(
            user_id="user123",
            org_id="org456",
            role=MembershipRole.ADMIN,
            invited_by="owner789",
        )

        # Get user's organizations
        memberships = await manager.get_user_memberships("user123")

        # Switch context to another org
        context = await manager.switch_context(
            user_id="user123",
            org_id="org789",
        )

        # Check permission in current context
        can_invite = context.membership.has_permission("members.invite")
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_orgs_per_user: int = 10,
    ):
        """
        Initialize multi-org manager.

        Args:
            db_path: Path to SQLite database (None for in-memory)
            max_orgs_per_user: Maximum organizations a user can belong to
        """
        self.db_path = db_path or ":memory:"
        self.max_orgs_per_user = max_orgs_per_user
        self._lock = threading.Lock()
        self._contexts: Dict[str, OrgContext] = {}  # user_id -> current context
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS organization_memberships (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    org_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',
                    status TEXT NOT NULL DEFAULT 'active',
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    joined_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    invited_by TEXT,
                    invitation_token TEXT,
                    custom_permissions TEXT DEFAULT '[]',
                    display_name TEXT,
                    department TEXT,
                    title TEXT,
                    UNIQUE(user_id, org_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memberships_user
                ON organization_memberships(user_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memberships_org
                ON organization_memberships(org_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memberships_token
                ON organization_memberships(invitation_token)
            """)
            conn.commit()

    async def add_member(
        self,
        user_id: str,
        org_id: str,
        role: MembershipRole = MembershipRole.MEMBER,
        invited_by: Optional[str] = None,
        is_primary: bool = False,
        status: MembershipStatus = MembershipStatus.ACTIVE,
        invitation_token: Optional[str] = None,
        display_name: Optional[str] = None,
        department: Optional[str] = None,
        title: Optional[str] = None,
    ) -> OrganizationMembership:
        """
        Add a user to an organization.

        Args:
            user_id: User ID
            org_id: Organization ID
            role: Role in organization
            invited_by: User ID who invited
            is_primary: Whether this is user's primary org
            status: Membership status
            invitation_token: Token for pending invitation
            display_name: Display name in this org
            department: Department in org
            title: Title in org

        Returns:
            Created membership

        Raises:
            ValueError: If user already in org or at limit
        """
        # Check existing membership
        existing = await self.get_membership(user_id, org_id)
        if existing:
            raise ValueError(f"User {user_id} already member of org {org_id}")

        # Check org limit
        user_orgs = await self.get_user_memberships(user_id)
        if len(user_orgs) >= self.max_orgs_per_user:
            raise ValueError(
                f"User {user_id} already in {len(user_orgs)} organizations "
                f"(limit: {self.max_orgs_per_user})"
            )

        # If this is primary, unset any existing primary
        if is_primary:
            await self._unset_all_primary(user_id)

        # If no existing memberships, make this primary
        if not user_orgs:
            is_primary = True

        membership = OrganizationMembership(
            user_id=user_id,
            org_id=org_id,
            role=role,
            status=status,
            is_primary=is_primary,
            invited_by=invited_by,
            invitation_token=invitation_token,
            display_name=display_name,
            department=department,
            title=title,
        )

        await self._save_membership(membership)
        logger.info(f"Added user {user_id} to org {org_id} with role {role.value}")
        return membership

    async def remove_member(
        self,
        user_id: str,
        org_id: str,
        removed_by: Optional[str] = None,
    ) -> bool:
        """
        Remove a user from an organization.

        Args:
            user_id: User to remove
            org_id: Organization ID
            removed_by: User performing removal

        Returns:
            True if removed, False if not found
        """
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            return False

        # Prevent removing the last owner
        if membership.role == MembershipRole.OWNER:
            owners = await self._count_role_in_org(org_id, MembershipRole.OWNER)
            if owners <= 1:
                raise ValueError("Cannot remove the last owner from organization")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM organization_memberships WHERE user_id = ? AND org_id = ?",
                (user_id, org_id),
            )
            conn.commit()

        # Clear context if removed from current org
        if user_id in self._contexts:
            if self._contexts[user_id].org_id == org_id:
                del self._contexts[user_id]

        logger.info(f"Removed user {user_id} from org {org_id}")
        return True

    async def update_role(
        self,
        user_id: str,
        org_id: str,
        new_role: MembershipRole,
        updated_by: Optional[str] = None,
    ) -> OrganizationMembership:
        """
        Update a member's role.

        Args:
            user_id: User ID
            org_id: Organization ID
            new_role: New role to assign
            updated_by: User performing update

        Returns:
            Updated membership
        """
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            raise ValueError(f"No membership found for user {user_id} in org {org_id}")

        # Prevent demoting the last owner
        if membership.role == MembershipRole.OWNER and new_role != MembershipRole.OWNER:
            owners = await self._count_role_in_org(org_id, MembershipRole.OWNER)
            if owners <= 1:
                raise ValueError("Cannot demote the last owner")

        membership.role = new_role
        membership.updated_at = datetime.now(timezone.utc)
        await self._save_membership(membership)

        logger.info(f"Updated role for {user_id} in {org_id} to {new_role.value}")
        return membership

    async def get_membership(
        self,
        user_id: str,
        org_id: str,
    ) -> Optional[OrganizationMembership]:
        """Get membership for user in organization."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM organization_memberships
                WHERE user_id = ? AND org_id = ?
                """,
                (user_id, org_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_membership(row)

    async def get_user_memberships(
        self,
        user_id: str,
        include_pending: bool = False,
    ) -> List[OrganizationMembership]:
        """Get all organization memberships for a user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if include_pending:
                cursor = conn.execute(
                    """
                    SELECT * FROM organization_memberships
                    WHERE user_id = ?
                    ORDER BY is_primary DESC, joined_at ASC
                    """,
                    (user_id,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM organization_memberships
                    WHERE user_id = ? AND status = 'active'
                    ORDER BY is_primary DESC, joined_at ASC
                    """,
                    (user_id,),
                )

            return [self._row_to_membership(row) for row in cursor.fetchall()]

    async def get_org_members(
        self,
        org_id: str,
        role_filter: Optional[MembershipRole] = None,
    ) -> List[OrganizationMembership]:
        """Get all members of an organization."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if role_filter:
                cursor = conn.execute(
                    """
                    SELECT * FROM organization_memberships
                    WHERE org_id = ? AND role = ? AND status = 'active'
                    ORDER BY role, joined_at ASC
                    """,
                    (org_id, role_filter.value),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM organization_memberships
                    WHERE org_id = ? AND status = 'active'
                    ORDER BY role, joined_at ASC
                    """,
                    (org_id,),
                )

            return [self._row_to_membership(row) for row in cursor.fetchall()]

    async def set_primary_org(
        self,
        user_id: str,
        org_id: str,
    ) -> OrganizationMembership:
        """Set user's primary organization."""
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            raise ValueError(f"No membership found for user {user_id} in org {org_id}")

        if membership.status != MembershipStatus.ACTIVE:
            raise ValueError("Cannot set inactive membership as primary")

        # Unset all primary for this user
        await self._unset_all_primary(user_id)

        # Set this as primary
        membership.is_primary = True
        membership.updated_at = datetime.now(timezone.utc)
        await self._save_membership(membership)

        logger.info(f"Set primary org for {user_id} to {org_id}")
        return membership

    async def get_primary_org(
        self,
        user_id: str,
    ) -> Optional[OrganizationMembership]:
        """Get user's primary organization membership."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM organization_memberships
                WHERE user_id = ? AND is_primary = 1 AND status = 'active'
                """,
                (user_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_membership(row)

    async def switch_context(
        self,
        user_id: str,
        org_id: str,
        organization: Optional[Organization] = None,
    ) -> OrgContext:
        """
        Switch user's current organization context.

        Args:
            user_id: User ID
            org_id: Target organization ID
            organization: Organization object (optional, for caching)

        Returns:
            New organization context
        """
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            raise ValueError(f"User {user_id} not member of org {org_id}")

        if membership.status != MembershipStatus.ACTIVE:
            raise ValueError(f"Membership in org {org_id} is not active")

        # Get or create organization object
        if organization is None:
            # In real implementation, would fetch from database
            organization = Organization(id=org_id, name=f"Org {org_id}")

        # Update switch count
        switch_count = 0
        if user_id in self._contexts:
            switch_count = self._contexts[user_id].switch_count + 1

        context = OrgContext(
            user_id=user_id,
            org_id=org_id,
            membership=membership,
            organization=organization,
            switch_count=switch_count,
        )

        self._contexts[user_id] = context
        logger.debug(f"Switched context for {user_id} to org {org_id}")
        return context

    async def get_current_context(
        self,
        user_id: str,
    ) -> Optional[OrgContext]:
        """Get user's current organization context."""
        return self._contexts.get(user_id)

    async def check_permission(
        self,
        user_id: str,
        org_id: str,
        permission: str,
    ) -> bool:
        """
        Check if user has permission in organization.

        Args:
            user_id: User ID
            org_id: Organization ID
            permission: Permission string (e.g., "members.invite")

        Returns:
            True if user has permission
        """
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            return False
        return membership.has_permission(permission)

    async def get_user_permissions(
        self,
        user_id: str,
        org_id: str,
    ) -> Set[str]:
        """Get all permissions for user in organization."""
        membership = await self.get_membership(user_id, org_id)
        if not membership:
            return set()
        return membership.permissions

    async def accept_invitation(
        self,
        invitation_token: str,
        user_id: str,
    ) -> OrganizationMembership:
        """
        Accept an organization invitation.

        Args:
            invitation_token: Invitation token
            user_id: User accepting the invitation

        Returns:
            Activated membership
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM organization_memberships
                WHERE invitation_token = ? AND status = 'pending'
                """,
                (invitation_token,),
            )
            row = cursor.fetchone()

        if not row:
            raise ValueError("Invalid or expired invitation token")

        membership = self._row_to_membership(row)

        # Update membership
        membership.status = MembershipStatus.ACTIVE
        membership.invitation_token = None
        membership.user_id = user_id  # Link to actual user
        membership.updated_at = datetime.now(timezone.utc)

        await self._save_membership(membership)
        logger.info(f"User {user_id} accepted invitation to org {membership.org_id}")
        return membership

    async def get_member_count(self, org_id: str) -> Dict[str, int]:
        """Get member counts by role for organization."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT role, COUNT(*) as count
                FROM organization_memberships
                WHERE org_id = ? AND status = 'active'
                GROUP BY role
                """,
                (org_id,),
            )
            counts = {"total": 0}
            for row in cursor.fetchall():
                counts[row[0]] = row[1]
                counts["total"] += row[1]
            return counts

    async def get_cross_org_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's cross-organization activity."""
        memberships = await self.get_user_memberships(user_id)

        if not memberships:
            return {
                "total_organizations": 0,
                "roles": {},
                "primary_org_id": None,
            }

        roles: Dict[str, int] = {}
        primary_org_id = None

        for m in memberships:
            role = m.role.value
            roles[role] = roles.get(role, 0) + 1
            if m.is_primary:
                primary_org_id = m.org_id

        return {
            "total_organizations": len(memberships),
            "roles": roles,
            "primary_org_id": primary_org_id,
            "organizations": [
                {
                    "org_id": m.org_id,
                    "role": m.role.value,
                    "is_primary": m.is_primary,
                    "joined_at": m.joined_at.isoformat(),
                }
                for m in memberships
            ],
        }

    async def _unset_all_primary(self, user_id: str) -> None:
        """Unset primary flag for all user's memberships."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE organization_memberships
                SET is_primary = 0, updated_at = ?
                WHERE user_id = ?
                """,
                (datetime.now(timezone.utc).isoformat(), user_id),
            )
            conn.commit()

    async def _count_role_in_org(
        self,
        org_id: str,
        role: MembershipRole,
    ) -> int:
        """Count members with specific role in organization."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM organization_memberships
                WHERE org_id = ? AND role = ? AND status = 'active'
                """,
                (org_id, role.value),
            )
            return cursor.fetchone()[0]

    async def _save_membership(self, membership: OrganizationMembership) -> None:
        """Save membership to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO organization_memberships (
                    id, user_id, org_id, role, status, is_primary,
                    joined_at, updated_at, invited_by, invitation_token,
                    custom_permissions, display_name, department, title
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    membership.id,
                    membership.user_id,
                    membership.org_id,
                    membership.role.value,
                    membership.status.value,
                    1 if membership.is_primary else 0,
                    membership.joined_at.isoformat(),
                    membership.updated_at.isoformat(),
                    membership.invited_by,
                    membership.invitation_token,
                    json.dumps(membership.custom_permissions),
                    membership.display_name,
                    membership.department,
                    membership.title,
                ),
            )
            conn.commit()

    def _row_to_membership(self, row: sqlite3.Row) -> OrganizationMembership:
        """Convert database row to membership object."""
        return OrganizationMembership(
            id=row["id"],
            user_id=row["user_id"],
            org_id=row["org_id"],
            role=MembershipRole(row["role"]),
            status=MembershipStatus(row["status"]),
            is_primary=bool(row["is_primary"]),
            joined_at=datetime.fromisoformat(row["joined_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            invited_by=row["invited_by"],
            invitation_token=row["invitation_token"],
            custom_permissions=json.loads(row["custom_permissions"] or "[]"),
            display_name=row["display_name"],
            department=row["department"],
            title=row["title"],
        )


# Global manager instance
_manager: Optional[MultiOrgManager] = None


def get_multi_org_manager(db_path: Optional[str] = None) -> MultiOrgManager:
    """Get or create global multi-org manager."""
    global _manager
    if _manager is None:
        _manager = MultiOrgManager(db_path=db_path)
    return _manager


__all__ = [
    "MembershipRole",
    "MembershipStatus",
    "OrganizationMembership",
    "OrgContext",
    "MultiOrgManager",
    "ROLE_PERMISSIONS",
    "get_multi_org_manager",
]
