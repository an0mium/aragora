"""
RBAC Enforcer.

Handles permission checking and enforcement for the enterprise control plane.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .types import (
    Action,
    IsolationContext,
    Permission,
    ResourceType,
    Role,
    RoleAssignment,
    Scope,
    SYSTEM_ROLES,
)

logger = logging.getLogger(__name__)


class PermissionDeniedException(Exception):
    """Raised when permission is denied for an operation."""

    def __init__(
        self,
        message: str,
        actor_id: str,
        resource: ResourceType,
        action: Action,
        context: IsolationContext | None = None,
    ):
        super().__init__(message)
        self.actor_id = actor_id
        self.resource = resource
        self.action = action
        self.context = context


@dataclass
class RBACConfig:
    """Configuration for RBAC enforcement."""

    # Enforcement
    enabled: bool = True
    deny_by_default: bool = True
    log_all_checks: bool = True

    # Caching
    cache_ttl_seconds: int = 300
    max_cache_size: int = 10000

    # Audit
    log_denials: bool = True
    log_grants: bool = False


@dataclass
class PermissionCheckResult:
    """Result of a permission check."""

    granted: bool
    reason: str
    matching_role: str | None = None
    matching_permission: Permission | None = None
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "granted": self.granted,
            "reason": self.reason,
            "matching_role": self.matching_role,
            "matching_permission": (
                self.matching_permission.to_dict() if self.matching_permission else None
            ),
            "checked_at": self.checked_at.isoformat(),
        }


class RBACEnforcer:
    """
    Enforces role-based access control across the system.

    Supports hierarchical permission inheritance:
    - Global permissions apply everywhere
    - Organization permissions apply to all workspaces in the org
    - Workspace permissions apply only within that workspace
    """

    def __init__(self, config: RBACConfig | None = None):
        self.config = config or RBACConfig()

        # Role storage
        self._roles: dict[str, Role] = dict(SYSTEM_ROLES)
        self._role_assignments: dict[str, list[RoleAssignment]] = {}

        # Audit log
        self._audit_log: list[dict] = []

        # Permission cache: (actor_id, org_id, workspace_id) -> set of permissions
        self._permission_cache: dict[tuple[str, str | None, str | None], set[Permission]] = {}
        self._cache_timestamps: dict[tuple[str, str | None, str | None], datetime] = {}

    async def check(
        self,
        actor: str,
        resource: ResourceType,
        action: Action,
        context: IsolationContext | None = None,
        resource_context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Check if an actor has permission to perform an action.

        Args:
            actor: Actor ID (user or service)
            resource: Resource type being accessed
            action: Action being performed
            context: Isolation context with org/workspace scope
            resource_context: Additional context for conditional permissions

        Returns:
            True if permission is granted
        """
        if not self.config.enabled:
            return True

        result = await self._check_permission(actor, resource, action, context, resource_context)

        # Log the check
        if self.config.log_all_checks:
            await self._log_check(actor, resource, action, context, result)

        return result.granted

    async def require(
        self,
        actor: str,
        resource: ResourceType,
        action: Action,
        context: IsolationContext | None = None,
        resource_context: dict[str, Any] | None = None,
    ) -> None:
        """
        Require permission, raising PermissionDeniedException if not granted.

        Args:
            actor: Actor ID (user or service)
            resource: Resource type being accessed
            action: Action being performed
            context: Isolation context with org/workspace scope
            resource_context: Additional context for conditional permissions

        Raises:
            PermissionDeniedException: If permission is denied
        """
        result = await self._check_permission(actor, resource, action, context, resource_context)

        # Log the check
        if self.config.log_all_checks or (not result.granted and self.config.log_denials):
            await self._log_check(actor, resource, action, context, result)

        if not result.granted:
            raise PermissionDeniedException(
                message=result.reason,
                actor_id=actor,
                resource=resource,
                action=action,
                context=context,
            )

    async def _check_permission(
        self,
        actor: str,
        resource: ResourceType,
        action: Action,
        context: IsolationContext | None,
        resource_context: dict[str, Any] | None,
    ) -> PermissionCheckResult:
        """Internal permission check logic."""
        org_id = context.organization_id if context else None
        workspace_id = context.workspace_id if context else None

        # Get effective permissions for actor
        permissions = await self._get_effective_permissions(actor, org_id, workspace_id)

        # Check if any permission grants access
        for perm in permissions:
            if perm.matches(resource, action, resource_context):
                return PermissionCheckResult(
                    granted=True,
                    reason="Permission granted",
                    matching_permission=perm,
                )

        # Check for ADMIN permission on the resource (grants all actions)
        for perm in permissions:
            if perm.resource == resource and perm.action == Action.ADMIN:
                return PermissionCheckResult(
                    granted=True,
                    reason="Admin permission grants all actions",
                    matching_permission=perm,
                )

        # Default deny
        if self.config.deny_by_default:
            return PermissionCheckResult(
                granted=False,
                reason=f"No permission grants {action.value} on {resource.value}",
            )

        return PermissionCheckResult(
            granted=True,
            reason="Default allow (deny_by_default=False)",
        )

    async def _get_effective_permissions(
        self,
        actor: str,
        org_id: str | None,
        workspace_id: str | None,
    ) -> set[Permission]:
        """Get all effective permissions for an actor in a scope."""
        cache_key = (actor, org_id, workspace_id)

        # Check cache
        if cache_key in self._permission_cache:
            cached_time = self._cache_timestamps.get(cache_key)
            if cached_time:
                age = (datetime.now(timezone.utc) - cached_time).total_seconds()
                if age < self.config.cache_ttl_seconds:
                    return self._permission_cache[cache_key]

        # Build effective permissions
        permissions: set[Permission] = set()

        assignments = self._role_assignments.get(actor, [])

        for assignment in assignments:
            if assignment.is_expired:
                continue

            # Check scope matching
            if assignment.scope == Scope.GLOBAL:
                # Global roles always apply
                pass
            elif assignment.scope == Scope.ORGANIZATION:
                # Org roles apply if we're in that org
                if org_id and assignment.scope_id != org_id:
                    continue
            elif assignment.scope == Scope.WORKSPACE:
                # Workspace roles only apply in that workspace
                if workspace_id and assignment.scope_id != workspace_id:
                    continue

            # Get role and add permissions
            role = self._roles.get(assignment.role_id)
            if role:
                permissions.update(role.permissions)

        # Cache result
        self._permission_cache[cache_key] = permissions
        self._cache_timestamps[cache_key] = datetime.now(timezone.utc)

        # Trim cache if too large
        if len(self._permission_cache) > self.config.max_cache_size:
            oldest_key = min(self._cache_timestamps, key=lambda k: self._cache_timestamps[k])
            del self._permission_cache[oldest_key]
            del self._cache_timestamps[oldest_key]

        return permissions

    async def assign_role(
        self,
        actor_id: str,
        role_id: str,
        scope: Scope,
        scope_id: str,
        assigned_by: str,
        expires_at: datetime | None = None,
    ) -> RoleAssignment:
        """
        Assign a role to an actor.

        Args:
            actor_id: User or service to assign role to
            role_id: Role to assign
            scope: Scope level (global, org, workspace)
            scope_id: ID of the scope (org_id or workspace_id)
            assigned_by: Who is making the assignment
            expires_at: Optional expiration time

        Returns:
            Created role assignment
        """
        # Validate role exists
        if role_id not in self._roles:
            raise ValueError(f"Role not found: {role_id}")

        assignment = RoleAssignment(
            actor_id=actor_id,
            role_id=role_id,
            scope=scope,
            scope_id=scope_id,
            assigned_by=assigned_by,
            expires_at=expires_at,
        )

        if actor_id not in self._role_assignments:
            self._role_assignments[actor_id] = []

        self._role_assignments[actor_id].append(assignment)

        # Invalidate cache
        self._invalidate_cache(actor_id)

        logger.info(f"Assigned role {role_id} to {actor_id} in {scope.value}:{scope_id}")

        return assignment

    async def revoke_role(
        self,
        actor_id: str,
        role_id: str,
        scope_id: str,
    ) -> bool:
        """
        Revoke a role from an actor.

        Args:
            actor_id: User or service to revoke from
            role_id: Role to revoke
            scope_id: Scope where role was assigned

        Returns:
            True if role was revoked
        """
        if actor_id not in self._role_assignments:
            return False

        initial_count = len(self._role_assignments[actor_id])

        self._role_assignments[actor_id] = [
            a
            for a in self._role_assignments[actor_id]
            if not (a.role_id == role_id and a.scope_id == scope_id)
        ]

        # Invalidate cache
        self._invalidate_cache(actor_id)

        revoked = len(self._role_assignments[actor_id]) < initial_count

        if revoked:
            logger.info(f"Revoked role {role_id} from {actor_id} in {scope_id}")

        return revoked

    async def get_actor_roles(
        self,
        actor_id: str,
        scope: Scope | None = None,
        scope_id: str | None = None,
    ) -> list[RoleAssignment]:
        """Get role assignments for an actor."""
        assignments = self._role_assignments.get(actor_id, [])

        if scope:
            assignments = [a for a in assignments if a.scope == scope]

        if scope_id:
            assignments = [a for a in assignments if a.scope_id == scope_id]

        return [a for a in assignments if not a.is_expired]

    async def create_role(
        self,
        role_id: str,
        name: str,
        description: str,
        permissions: set[Permission],
        scope: Scope,
        created_by: str,
    ) -> Role:
        """Create a custom role."""
        if role_id in self._roles:
            raise ValueError(f"Role already exists: {role_id}")

        role = Role(
            id=role_id,
            name=name,
            description=description,
            permissions=permissions,
            scope=scope,
            is_system=False,
            created_by=created_by,
        )

        self._roles[role_id] = role

        logger.info(f"Created role {role_id}")

        return role

    async def delete_role(self, role_id: str) -> bool:
        """Delete a custom role."""
        role = self._roles.get(role_id)
        if not role:
            return False

        if role.is_system:
            raise ValueError(f"Cannot delete system role: {role_id}")

        del self._roles[role_id]

        # Remove all assignments of this role
        for actor_id in self._role_assignments:
            self._role_assignments[actor_id] = [
                a for a in self._role_assignments[actor_id] if a.role_id != role_id
            ]
            self._invalidate_cache(actor_id)

        logger.info(f"Deleted role {role_id}")

        return True

    async def get_role(self, role_id: str) -> Role | None:
        """Get a role by ID."""
        return self._roles.get(role_id)

    async def list_roles(self, scope: Scope | None = None) -> list[Role]:
        """List all roles, optionally filtered by scope."""
        roles = list(self._roles.values())
        if scope:
            roles = [r for r in roles if r.scope == scope]
        return roles

    def _invalidate_cache(self, actor_id: str) -> None:
        """Invalidate permission cache for an actor."""
        keys_to_remove = [k for k in self._permission_cache if k[0] == actor_id]
        for key in keys_to_remove:
            del self._permission_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]

    async def _log_check(
        self,
        actor: str,
        resource: ResourceType,
        action: Action,
        context: IsolationContext | None,
        result: PermissionCheckResult,
    ) -> None:
        """Log a permission check."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "resource": resource.value,
            "action": action.value,
            "org_id": context.organization_id if context else None,
            "workspace_id": context.workspace_id if context else None,
            "granted": result.granted,
            "reason": result.reason,
        }

        self._audit_log.append(entry)

        # Trim old entries
        max_entries = 10000
        if len(self._audit_log) > max_entries:
            self._audit_log = self._audit_log[-max_entries:]

        # Log to standard logger
        level = logging.DEBUG if result.granted else logging.WARNING
        logger.log(
            level,
            f"RBAC: {actor} {action.value} {resource.value} -> "
            f"{'GRANTED' if result.granted else 'DENIED'} ({result.reason})",
        )

    async def populate_context_permissions(
        self,
        context: IsolationContext,
    ) -> IsolationContext:
        """
        Populate an IsolationContext with effective permissions.

        This is used at request entry to cache permissions for the request.
        """
        permissions = await self._get_effective_permissions(
            context.actor_id,
            context.organization_id,
            context.workspace_id,
        )

        assignments = await self.get_actor_roles(context.actor_id)

        # Update context with cached data
        context._effective_permissions = permissions
        context._role_assignments = assignments

        return context


# Global instance
_rbac_enforcer: RBACEnforcer | None = None


def get_rbac_enforcer(config: RBACConfig | None = None) -> RBACEnforcer:
    """Get or create the global RBAC enforcer."""
    global _rbac_enforcer
    if _rbac_enforcer is None:
        _rbac_enforcer = RBACEnforcer(config)
    return _rbac_enforcer


__all__ = [
    "RBACEnforcer",
    "RBACConfig",
    "PermissionCheckResult",
    "PermissionDeniedException",
    "get_rbac_enforcer",
]
