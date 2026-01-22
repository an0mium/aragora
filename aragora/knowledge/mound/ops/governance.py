"""
Governance Module for Knowledge Mound.

Provides enterprise governance features:
- Role-Based Access Control (RBAC)
- Audit trail logging
- Policy enforcement
- Compliance tracking

Phase A2 - Workspace Governance
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Role-Based Access Control
# =============================================================================


class Permission(str, Enum):
    """Knowledge Mound permissions."""

    # Item permissions
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

    # Workspace permissions
    MANAGE_WORKSPACE = "manage_workspace"
    SHARE = "share"
    EXPORT = "export"

    # Admin permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    VIEW_AUDIT = "view_audit"
    MANAGE_POLICIES = "manage_policies"

    # System permissions
    ADMIN = "admin"  # Full access


class BuiltinRole(str, Enum):
    """Built-in role templates."""

    VIEWER = "viewer"  # Read-only access
    CONTRIBUTOR = "contributor"  # Read + create
    EDITOR = "editor"  # Read + create + update
    MANAGER = "manager"  # Full item access + sharing
    ADMIN = "admin"  # Full access including user management


@dataclass
class Role:
    """A role with associated permissions."""

    id: str
    name: str
    description: str
    permissions: Set[Permission]
    workspace_id: Optional[str] = None  # None = global role
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    is_builtin: bool = False

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": [p.value for p in self.permissions],
            "workspace_id": self.workspace_id,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "is_builtin": self.is_builtin,
        }


@dataclass
class RoleAssignment:
    """Assignment of a role to a user."""

    id: str
    user_id: str
    role_id: str
    workspace_id: Optional[str] = None  # None = global assignment
    assigned_at: datetime = field(default_factory=datetime.now)
    assigned_by: Optional[str] = None
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if assignment has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "role_id": self.role_id,
            "workspace_id": self.workspace_id,
            "assigned_at": self.assigned_at.isoformat(),
            "assigned_by": self.assigned_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


# Built-in role definitions
BUILTIN_ROLES: Dict[BuiltinRole, Role] = {
    BuiltinRole.VIEWER: Role(
        id="builtin:viewer",
        name="Viewer",
        description="Read-only access to knowledge items",
        permissions={Permission.READ},
        is_builtin=True,
    ),
    BuiltinRole.CONTRIBUTOR: Role(
        id="builtin:contributor",
        name="Contributor",
        description="Can read and create knowledge items",
        permissions={Permission.READ, Permission.CREATE},
        is_builtin=True,
    ),
    BuiltinRole.EDITOR: Role(
        id="builtin:editor",
        name="Editor",
        description="Can read, create, and update knowledge items",
        permissions={Permission.READ, Permission.CREATE, Permission.UPDATE},
        is_builtin=True,
    ),
    BuiltinRole.MANAGER: Role(
        id="builtin:manager",
        name="Manager",
        description="Full item access plus sharing and workspace management",
        permissions={
            Permission.READ,
            Permission.CREATE,
            Permission.UPDATE,
            Permission.DELETE,
            Permission.SHARE,
            Permission.EXPORT,
            Permission.MANAGE_WORKSPACE,
        },
        is_builtin=True,
    ),
    BuiltinRole.ADMIN: Role(
        id="builtin:admin",
        name="Administrator",
        description="Full access to all features",
        permissions={Permission.ADMIN},
        is_builtin=True,
    ),
}


class RBACManager:
    """Manages role-based access control."""

    def __init__(self):
        """Initialize RBAC manager."""
        self._roles: Dict[str, Role] = {}
        self._assignments: Dict[str, RoleAssignment] = {}
        self._user_roles: Dict[str, Set[str]] = {}  # user_id -> role_ids
        self._lock = asyncio.Lock()

        # Initialize builtin roles
        for role in BUILTIN_ROLES.values():
            self._roles[role.id] = role

    async def create_role(
        self,
        name: str,
        permissions: Set[Permission],
        description: str = "",
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Role:
        """Create a new custom role.

        Args:
            name: Role name
            permissions: Set of permissions
            description: Role description
            workspace_id: Optional workspace scope
            created_by: User creating the role

        Returns:
            Created Role
        """
        import uuid

        role = Role(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            permissions=permissions,
            workspace_id=workspace_id,
            created_by=created_by,
        )

        async with self._lock:
            self._roles[role.id] = role

        logger.info(f"Created role: {role.name} ({role.id})")
        return role

    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        workspace_id: Optional[str] = None,
        assigned_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> RoleAssignment:
        """Assign a role to a user.

        Args:
            user_id: User to assign role to
            role_id: Role to assign
            workspace_id: Optional workspace scope
            assigned_by: User making the assignment
            expires_at: Optional expiration

        Returns:
            RoleAssignment
        """
        import uuid

        async with self._lock:
            if role_id not in self._roles:
                raise ValueError(f"Role not found: {role_id}")

            assignment = RoleAssignment(
                id=str(uuid.uuid4()),
                user_id=user_id,
                role_id=role_id,
                workspace_id=workspace_id,
                assigned_by=assigned_by,
                expires_at=expires_at,
            )

            self._assignments[assignment.id] = assignment

            if user_id not in self._user_roles:
                self._user_roles[user_id] = set()
            self._user_roles[user_id].add(role_id)

        logger.info(f"Assigned role {role_id} to user {user_id}")
        return assignment

    async def revoke_role(
        self,
        user_id: str,
        role_id: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """Revoke a role from a user.

        Args:
            user_id: User to revoke from
            role_id: Role to revoke
            workspace_id: Optional workspace scope

        Returns:
            True if revoked, False if not found
        """
        async with self._lock:
            # Find matching assignment
            to_remove = None
            for aid, assignment in self._assignments.items():
                if (
                    assignment.user_id == user_id
                    and assignment.role_id == role_id
                    and assignment.workspace_id == workspace_id
                ):
                    to_remove = aid
                    break

            if to_remove:
                del self._assignments[to_remove]
                if user_id in self._user_roles:
                    self._user_roles[user_id].discard(role_id)
                logger.info(f"Revoked role {role_id} from user {user_id}")
                return True

            return False

    async def check_permission(
        self,
        user_id: str,
        permission: Permission,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """Check if user has a permission.

        Args:
            user_id: User to check
            permission: Permission to check
            workspace_id: Optional workspace context

        Returns:
            True if user has permission
        """
        async with self._lock:
            role_ids = self._user_roles.get(user_id, set())

            for role_id in role_ids:
                role = self._roles.get(role_id)
                if not role:
                    continue

                # Check workspace scope
                if workspace_id and role.workspace_id:
                    if role.workspace_id != workspace_id:
                        continue

                if role.has_permission(permission):
                    return True

            return False

    async def get_user_permissions(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> Set[Permission]:
        """Get all permissions for a user.

        Args:
            user_id: User to get permissions for
            workspace_id: Optional workspace filter

        Returns:
            Set of permissions
        """
        permissions: Set[Permission] = set()

        async with self._lock:
            role_ids = self._user_roles.get(user_id, set())

            for role_id in role_ids:
                role = self._roles.get(role_id)
                if not role:
                    continue

                if workspace_id and role.workspace_id:
                    if role.workspace_id != workspace_id:
                        continue

                if Permission.ADMIN in role.permissions:
                    return set(Permission)  # All permissions

                permissions.update(role.permissions)

        return permissions

    async def get_user_roles(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> List[Role]:
        """Get all roles for a user."""
        async with self._lock:
            role_ids = self._user_roles.get(user_id, set())
            roles = []

            for role_id in role_ids:
                role = self._roles.get(role_id)
                if role:
                    if workspace_id and role.workspace_id:
                        if role.workspace_id == workspace_id:
                            roles.append(role)
                    else:
                        roles.append(role)

            return roles


# =============================================================================
# Audit Trail
# =============================================================================


class AuditAction(str, Enum):
    """Auditable actions in Knowledge Mound."""

    # Item actions
    ITEM_CREATE = "item.create"
    ITEM_READ = "item.read"
    ITEM_UPDATE = "item.update"
    ITEM_DELETE = "item.delete"

    # Sharing actions
    SHARE_GRANT = "share.grant"
    SHARE_REVOKE = "share.revoke"

    # Admin actions
    ROLE_CREATE = "role.create"
    ROLE_ASSIGN = "role.assign"
    ROLE_REVOKE = "role.revoke"

    # Policy actions
    POLICY_CREATE = "policy.create"
    POLICY_UPDATE = "policy.update"
    POLICY_DELETE = "policy.delete"

    # System actions
    EXPORT = "export"
    IMPORT = "import"
    BULK_DELETE = "bulk_delete"


@dataclass
class AuditEntry:
    """An entry in the audit trail."""

    id: str
    action: AuditAction
    actor_id: str  # User who performed action
    resource_type: str  # Type of resource affected
    resource_id: str  # ID of resource affected
    workspace_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "action": self.action.value,
            "actor_id": self.actor_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "workspace_id": self.workspace_id,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "success": self.success,
            "error_message": self.error_message,
        }


class AuditTrail:
    """Manages audit trail logging."""

    def __init__(self, max_entries: int = 100000):
        """Initialize audit trail.

        Args:
            max_entries: Maximum entries to keep in memory
        """
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def log(
        self,
        action: AuditAction,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        workspace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> AuditEntry:
        """Log an audit entry.

        Args:
            action: Action performed
            actor_id: User who performed action
            resource_type: Type of resource
            resource_id: ID of resource
            workspace_id: Optional workspace
            details: Additional details
            ip_address: Client IP
            user_agent: Client user agent
            success: Whether action succeeded
            error_message: Error if failed

        Returns:
            Created AuditEntry
        """
        import uuid

        entry = AuditEntry(
            id=str(uuid.uuid4()),
            action=action,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=workspace_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
        )

        async with self._lock:
            self._entries.append(entry)

            # Trim if over limit
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

        logger.debug(f"Audit: {action.value} by {actor_id} on {resource_type}/{resource_id}")

        return entry

    async def query(
        self,
        actor_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """Query audit entries.

        Args:
            actor_id: Filter by actor
            action: Filter by action
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            workspace_id: Filter by workspace
            start_time: Filter by start time
            end_time: Filter by end time
            success_only: Only successful actions
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching entries
        """
        async with self._lock:
            results = self._entries

            if actor_id:
                results = [e for e in results if e.actor_id == actor_id]

            if action:
                results = [e for e in results if e.action == action]

            if resource_type:
                results = [e for e in results if e.resource_type == resource_type]

            if resource_id:
                results = [e for e in results if e.resource_id == resource_id]

            if workspace_id:
                results = [e for e in results if e.workspace_id == workspace_id]

            if start_time:
                results = [e for e in results if e.timestamp >= start_time]

            if end_time:
                results = [e for e in results if e.timestamp <= end_time]

            if success_only:
                results = [e for e in results if e.success]

            # Sort by timestamp descending
            results = sorted(results, key=lambda e: e.timestamp, reverse=True)

            return results[offset : offset + limit]

    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get activity summary for a user.

        Args:
            user_id: User to get activity for
            days: Number of days to look back

        Returns:
            Activity summary
        """
        from datetime import timedelta

        start_time = datetime.now() - timedelta(days=days)

        entries = await self.query(
            actor_id=user_id,
            start_time=start_time,
        )

        by_action: Dict[str, int] = {}
        by_resource: Dict[str, int] = {}

        for entry in entries:
            by_action[entry.action.value] = by_action.get(entry.action.value, 0) + 1
            by_resource[entry.resource_type] = by_resource.get(entry.resource_type, 0) + 1

        return {
            "user_id": user_id,
            "period_days": days,
            "total_actions": len(entries),
            "by_action": by_action,
            "by_resource": by_resource,
            "success_rate": sum(1 for e in entries if e.success) / len(entries) if entries else 1.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get audit trail statistics."""
        by_action: Dict[str, int] = {}
        by_resource: Dict[str, int] = {}
        failures = 0

        for entry in self._entries:
            by_action[entry.action.value] = by_action.get(entry.action.value, 0) + 1
            by_resource[entry.resource_type] = by_resource.get(entry.resource_type, 0) + 1
            if not entry.success:
                failures += 1

        return {
            "total_entries": len(self._entries),
            "by_action": by_action,
            "by_resource": by_resource,
            "failures": failures,
            "success_rate": (len(self._entries) - failures) / len(self._entries)
            if self._entries
            else 1.0,
        }


# =============================================================================
# Governance Mixin
# =============================================================================


class GovernanceMixin:
    """Mixin for governance operations on KnowledgeMound."""

    _rbac_manager: Optional[RBACManager] = None
    _audit_trail: Optional[AuditTrail] = None

    def _get_rbac_manager(self) -> RBACManager:
        """Get or create RBAC manager."""
        if self._rbac_manager is None:
            self._rbac_manager = RBACManager()
        return self._rbac_manager

    def _get_audit_trail(self) -> AuditTrail:
        """Get or create audit trail."""
        if self._audit_trail is None:
            self._audit_trail = AuditTrail()
        return self._audit_trail

    # RBAC methods
    async def create_role(
        self,
        name: str,
        permissions: Set[Permission],
        description: str = "",
        workspace_id: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Role:
        """Create a new role."""
        return await self._get_rbac_manager().create_role(
            name, permissions, description, workspace_id, created_by
        )

    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        workspace_id: Optional[str] = None,
        assigned_by: Optional[str] = None,
    ) -> RoleAssignment:
        """Assign a role to a user."""
        return await self._get_rbac_manager().assign_role(
            user_id, role_id, workspace_id, assigned_by
        )

    async def revoke_role(
        self,
        user_id: str,
        role_id: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """Revoke a role from a user."""
        return await self._get_rbac_manager().revoke_role(user_id, role_id, workspace_id)

    async def check_permission(
        self,
        user_id: str,
        permission: Permission,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """Check if user has a permission."""
        return await self._get_rbac_manager().check_permission(user_id, permission, workspace_id)

    async def get_user_permissions(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> Set[Permission]:
        """Get all permissions for a user."""
        return await self._get_rbac_manager().get_user_permissions(user_id, workspace_id)

    # Audit methods
    async def log_audit(
        self,
        action: AuditAction,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        workspace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> AuditEntry:
        """Log an audit entry."""
        return await self._get_audit_trail().log(
            action=action,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=workspace_id,
            details=details,
            success=success,
        )

    async def query_audit(
        self,
        actor_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Query audit entries."""
        return await self._get_audit_trail().query(
            actor_id=actor_id,
            action=action,
            workspace_id=workspace_id,
            limit=limit,
        )

    async def get_user_activity(
        self,
        user_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get activity summary for a user."""
        return await self._get_audit_trail().get_user_activity(user_id, days)

    def get_governance_stats(self) -> Dict[str, Any]:
        """Get governance statistics."""
        return {
            "audit": self._get_audit_trail().get_stats(),
        }


# Singleton instances
_rbac_manager: Optional[RBACManager] = None
_audit_trail: Optional[AuditTrail] = None


def get_rbac_manager() -> RBACManager:
    """Get the global RBAC manager instance."""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def get_audit_trail() -> AuditTrail:
    """Get the global audit trail instance."""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail
