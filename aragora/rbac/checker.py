"""
RBAC Permission Checker - Evaluates access control decisions.

Provides the core logic for checking if a user has permission to perform
an action on a resource, with support for caching and audit logging.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

from .defaults import SYSTEM_PERMISSIONS, SYSTEM_ROLES, get_role_permissions
from .models import (
    Action,
    AuthorizationContext,
    AuthorizationDecision,
    ResourceType,
    RoleAssignment,
)

if TYPE_CHECKING:
    from .audit import AuthorizationAuditor

logger = logging.getLogger(__name__)


class PermissionChecker:
    """
    Core permission checking service.

    Evaluates whether a user has permission to perform actions on resources,
    with support for:
    - Role-based permissions
    - Permission inheritance
    - Resource-level access control
    - API key scopes
    - Caching for performance
    - Audit logging
    """

    def __init__(
        self,
        auditor: AuthorizationAuditor | None = None,
        cache_ttl: int = 300,
        enable_cache: bool = True,
    ) -> None:
        """
        Initialize the permission checker.

        Args:
            auditor: Optional auditor for logging authorization decisions
            cache_ttl: Cache TTL in seconds (default 5 minutes)
            enable_cache: Whether to enable decision caching
        """
        self._auditor = auditor
        self._cache_ttl = cache_ttl
        self._enable_cache = enable_cache
        self._decision_cache: dict[str, tuple[AuthorizationDecision, datetime]] = {}
        self._role_assignments: dict[str, list[RoleAssignment]] = {}
        self._custom_roles: dict[str, dict[str, Any]] = {}
        self._resource_policies: dict[str, Callable[..., bool]] = {}

    def check_permission(
        self,
        context: AuthorizationContext,
        permission_key: str,
        resource_id: str | None = None,
    ) -> AuthorizationDecision:
        """
        Check if the context has a specific permission.

        Args:
            context: Authorization context (user, roles, etc.)
            permission_key: Permission to check (e.g., "debates.create")
            resource_id: Optional specific resource ID

        Returns:
            AuthorizationDecision with result and reason
        """
        # Check cache first
        cache_key = self._cache_key(context, permission_key, resource_id)
        if self._enable_cache:
            cached = self._get_cached_decision(cache_key)
            if cached:
                return cached

        # Resolve permissions if not already done
        if not context.permissions:
            context = self._resolve_permissions(context)

        # Check permission
        decision = self._evaluate_permission(context, permission_key, resource_id)

        # Cache the decision
        if self._enable_cache:
            self._cache_decision(cache_key, decision)

        # Audit log
        if self._auditor:
            self._auditor.log_decision(decision)

        return decision

    def check_resource_access(
        self,
        context: AuthorizationContext,
        resource_type: ResourceType,
        action: Action,
        resource_id: str,
        resource_attrs: dict[str, Any] | None = None,
    ) -> AuthorizationDecision:
        """
        Check if context can perform an action on a specific resource.

        This method supports attribute-based access control (ABAC) by
        allowing resource policies to be evaluated.

        Args:
            context: Authorization context
            resource_type: Type of resource
            action: Action to perform
            resource_id: Specific resource ID
            resource_attrs: Optional resource attributes for ABAC

        Returns:
            AuthorizationDecision
        """
        permission_key = f"{resource_type.value}.{action.value}"

        # First check base permission
        decision = self.check_permission(context, permission_key, resource_id)
        if not decision.allowed:
            return decision

        # Check resource-specific policies
        policy_key = f"{resource_type.value}:{resource_id}"
        if policy_key in self._resource_policies:
            policy = self._resource_policies[policy_key]
            if not policy(context, action, resource_attrs or {}):
                return AuthorizationDecision(
                    allowed=False,
                    reason=f"Resource policy denied access to {resource_id}",
                    permission_key=permission_key,
                    resource_id=resource_id,
                    context=context,
                )

        # Check API key scope for resource access
        if context.api_key_scope:
            if not context.api_key_scope.allows_resource(resource_type, resource_id):
                return AuthorizationDecision(
                    allowed=False,
                    reason=f"API key scope does not allow access to {resource_id}",
                    permission_key=permission_key,
                    resource_id=resource_id,
                    context=context,
                )

        return decision

    def has_role(self, context: AuthorizationContext, role_name: str) -> bool:
        """Check if context has a specific role."""
        return role_name in context.roles

    def has_any_role(self, context: AuthorizationContext, *role_names: str) -> bool:
        """Check if context has any of the specified roles."""
        return bool(context.roles & set(role_names))

    def has_all_roles(self, context: AuthorizationContext, *role_names: str) -> bool:
        """Check if context has all of the specified roles."""
        return set(role_names) <= context.roles

    def is_owner(self, context: AuthorizationContext) -> bool:
        """Check if context has owner role."""
        return self.has_role(context, "owner")

    def is_admin(self, context: AuthorizationContext) -> bool:
        """Check if context has admin or owner role."""
        return self.has_any_role(context, "owner", "admin")

    def register_resource_policy(
        self,
        resource_type: ResourceType,
        resource_id: str,
        policy: Callable[[AuthorizationContext, Action, dict[str, Any]], bool],
    ) -> None:
        """
        Register a custom policy for a specific resource.

        Args:
            resource_type: Type of resource
            resource_id: Specific resource ID
            policy: Policy function that returns True if access is allowed
        """
        policy_key = f"{resource_type.value}:{resource_id}"
        self._resource_policies[policy_key] = policy

    def add_role_assignment(self, assignment: RoleAssignment) -> None:
        """Add a role assignment to the checker's cache."""
        if assignment.user_id not in self._role_assignments:
            self._role_assignments[assignment.user_id] = []
        self._role_assignments[assignment.user_id].append(assignment)

    def remove_role_assignment(self, user_id: str, role_id: str, org_id: str | None = None) -> None:
        """Remove a role assignment."""
        if user_id in self._role_assignments:
            self._role_assignments[user_id] = [
                a for a in self._role_assignments[user_id]
                if not (a.role_id == role_id and a.org_id == org_id)
            ]

    def get_user_roles(self, user_id: str, org_id: str | None = None) -> set[str]:
        """Get all active roles for a user in an organization."""
        roles: set[str] = set()
        assignments = self._role_assignments.get(user_id, [])

        for assignment in assignments:
            if not assignment.is_valid:
                continue
            if org_id is None or assignment.org_id == org_id:
                roles.add(assignment.role_id)

        return roles

    def clear_cache(self, user_id: str | None = None) -> None:
        """Clear decision cache, optionally for a specific user."""
        if user_id:
            keys_to_remove = [k for k in self._decision_cache if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self._decision_cache[key]
        else:
            self._decision_cache.clear()

    def _resolve_permissions(self, context: AuthorizationContext) -> AuthorizationContext:
        """Resolve all permissions from roles."""
        all_permissions: set[str] = set()

        for role_name in context.roles:
            # Get system role permissions
            permissions = get_role_permissions(role_name, include_inherited=True)
            all_permissions |= permissions

            # Check for custom role
            if context.org_id:
                custom_key = f"{context.org_id}:{role_name}"
                if custom_key in self._custom_roles:
                    custom = self._custom_roles[custom_key]
                    all_permissions |= custom.get("permissions", set())

        # Create new context with resolved permissions
        return AuthorizationContext(
            user_id=context.user_id,
            org_id=context.org_id,
            roles=context.roles,
            permissions=all_permissions,
            api_key_scope=context.api_key_scope,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            request_id=context.request_id,
            timestamp=context.timestamp,
        )

    def _evaluate_permission(
        self,
        context: AuthorizationContext,
        permission_key: str,
        resource_id: str | None,
    ) -> AuthorizationDecision:
        """Evaluate if context has permission."""
        # Check for exact permission
        if permission_key in context.permissions:
            return AuthorizationDecision(
                allowed=True,
                reason=f"Permission '{permission_key}' granted",
                permission_key=permission_key,
                resource_id=resource_id,
                context=context,
            )

        # Check for wildcard permission (e.g., "debates.*")
        resource = permission_key.split(".")[0]
        wildcard_key = f"{resource}.*"
        if wildcard_key in context.permissions:
            return AuthorizationDecision(
                allowed=True,
                reason=f"Wildcard permission '{wildcard_key}' grants '{permission_key}'",
                permission_key=permission_key,
                resource_id=resource_id,
                context=context,
            )

        # Check for super wildcard
        if "*" in context.permissions:
            return AuthorizationDecision(
                allowed=True,
                reason="Super wildcard '*' grants all permissions",
                permission_key=permission_key,
                resource_id=resource_id,
                context=context,
            )

        # Check API key scope
        if context.api_key_scope:
            if not context.api_key_scope.allows_permission(permission_key):
                return AuthorizationDecision(
                    allowed=False,
                    reason=f"API key scope does not include '{permission_key}'",
                    permission_key=permission_key,
                    resource_id=resource_id,
                    context=context,
                )

        # Permission denied
        return AuthorizationDecision(
            allowed=False,
            reason=f"Permission '{permission_key}' not granted to user",
            permission_key=permission_key,
            resource_id=resource_id,
            context=context,
        )

    def _cache_key(
        self,
        context: AuthorizationContext,
        permission_key: str,
        resource_id: str | None,
    ) -> str:
        """Generate cache key for a permission check."""
        roles_hash = hash(frozenset(context.roles))
        return f"{context.user_id}:{context.org_id}:{roles_hash}:{permission_key}:{resource_id}"

    def _get_cached_decision(self, cache_key: str) -> AuthorizationDecision | None:
        """Get cached decision if valid."""
        if cache_key not in self._decision_cache:
            return None

        decision, cached_at = self._decision_cache[cache_key]
        age = (datetime.utcnow() - cached_at).total_seconds()

        if age > self._cache_ttl:
            del self._decision_cache[cache_key]
            return None

        # Return cached decision with cached flag
        return AuthorizationDecision(
            allowed=decision.allowed,
            reason=decision.reason,
            permission_key=decision.permission_key,
            resource_id=decision.resource_id,
            context=decision.context,
            checked_at=decision.checked_at,
            cached=True,
        )

    def _cache_decision(self, cache_key: str, decision: AuthorizationDecision) -> None:
        """Cache a decision."""
        self._decision_cache[cache_key] = (decision, datetime.utcnow())


# Global permission checker instance
_permission_checker: PermissionChecker | None = None


def get_permission_checker() -> PermissionChecker:
    """Get or create the global permission checker instance."""
    global _permission_checker
    if _permission_checker is None:
        _permission_checker = PermissionChecker()
    return _permission_checker


def set_permission_checker(checker: PermissionChecker) -> None:
    """Set the global permission checker instance."""
    global _permission_checker
    _permission_checker = checker


def check_permission(
    context: AuthorizationContext,
    permission_key: str,
    resource_id: str | None = None,
) -> AuthorizationDecision:
    """Convenience function to check permission using global checker."""
    return get_permission_checker().check_permission(context, permission_key, resource_id)


def has_permission(
    context: AuthorizationContext,
    permission_key: str,
    resource_id: str | None = None,
) -> bool:
    """Convenience function to check if permission is granted."""
    return check_permission(context, permission_key, resource_id).allowed
