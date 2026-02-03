"""
Knowledge Mound RBAC Access Control.

Provides permission checking for Knowledge Mound operations.
Integrates with Aragora's RBAC system to enforce:
- Tenant isolation (users can only access their tenant's knowledge)
- Role-based access (read, write, admin permissions)
- Resource-level permissions (per-item visibility)

Usage:
    from aragora.knowledge.mound.access import KnowledgeMoundAccessControl

    # Create access control instance
    ac = KnowledgeMoundAccessControl(checker)

    # Check permissions before operations
    if await ac.can_read(ctx, item_id):
        item = await km.get_item(item_id)

    # Use decorators for handler methods
    @ac.require_permission("knowledge:read")
    async def list_knowledge(ctx, ...):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, ParamSpec

if TYPE_CHECKING:
    from aragora.rbac.checker import PermissionChecker
    from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class KnowledgeAccessResult:
    """Result of a knowledge access check."""

    allowed: bool
    permission: str
    reason: str = ""
    tenant_id: str | None = None
    user_id: str | None = None
    resource_id: str | None = None
    checked_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "permission": self.permission,
            "reason": self.reason,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "resource_id": self.resource_id,
            "checked_at": self.checked_at.isoformat(),
        }


class KnowledgeMoundAccessControl:
    """RBAC access control for Knowledge Mound operations.

    Enforces permission checks and tenant isolation for all KM operations.
    Uses the central RBAC PermissionChecker for consistency.
    """

    # Permission mappings
    PERMISSION_READ = "knowledge:read"
    PERMISSION_WRITE = "knowledge:write"
    PERMISSION_UPDATE = "knowledge:update"
    PERMISSION_DELETE = "knowledge:delete"
    PERMISSION_SHARE = "knowledge:share"
    PERMISSION_ADMIN = "knowledge:admin"

    def __init__(
        self,
        checker: "PermissionChecker | None" = None,
        enforce_tenant_isolation: bool = True,
        audit_access: bool = True,
    ) -> None:
        """Initialize access control.

        Args:
            checker: RBAC permission checker instance
            enforce_tenant_isolation: Whether to enforce tenant boundaries
            audit_access: Whether to log access attempts
        """
        self._checker = checker
        self._enforce_tenant_isolation = enforce_tenant_isolation
        self._audit_access = audit_access
        self._access_log: list[KnowledgeAccessResult] = []

    @property
    def checker(self) -> "PermissionChecker | None":
        """Get the permission checker, initializing lazily if needed."""
        if self._checker is None:
            try:
                from aragora.rbac.checker import get_permission_checker

                self._checker = get_permission_checker()
            except ImportError:
                logger.debug("RBAC checker not available")
        return self._checker

    def _log_access(self, result: KnowledgeAccessResult) -> None:
        """Log an access check result."""
        if self._audit_access:
            self._access_log.append(result)
            # Keep log bounded
            if len(self._access_log) > 1000:
                self._access_log = self._access_log[-500:]

            log_level = logging.DEBUG if result.allowed else logging.WARNING
            logger.log(
                log_level,
                "km_access_check permission=%s allowed=%s user=%s tenant=%s resource=%s reason=%s",
                result.permission,
                result.allowed,
                result.user_id,
                result.tenant_id,
                result.resource_id,
                result.reason,
            )

    def _check_permission(
        self,
        ctx: "AuthorizationContext | None",
        permission: str,
        resource_id: str | None = None,
    ) -> KnowledgeAccessResult:
        """Check if context has permission.

        Args:
            ctx: Authorization context (user, tenant, roles)
            permission: Permission to check
            resource_id: Optional resource ID for resource-level checks

        Returns:
            KnowledgeAccessResult with the check outcome
        """
        if ctx is None:
            # No context - allow if RBAC not enforced
            if self.checker is None:
                return KnowledgeAccessResult(
                    allowed=True,
                    permission=permission,
                    reason="RBAC not configured, allowing access",
                )
            return KnowledgeAccessResult(
                allowed=False,
                permission=permission,
                reason="No authorization context provided",
            )

        user_id = getattr(ctx, "user_id", None)
        tenant_id = getattr(ctx, "tenant_id", None)

        # Check with RBAC checker
        checker = self.checker
        if checker is None:
            return KnowledgeAccessResult(
                allowed=True,
                permission=permission,
                reason="RBAC checker not available, allowing access",
                user_id=user_id,
                tenant_id=tenant_id,
                resource_id=resource_id,
            )

        # Use checker's has_permission method
        try:
            # Build permission string
            allowed = checker.has_permission(ctx, permission)
            reason = "Permission granted" if allowed else "Permission denied"
        except Exception as e:
            logger.warning("Permission check failed: %s", e)
            allowed = False
            reason = f"Permission check error: {e}"

        result = KnowledgeAccessResult(
            allowed=allowed,
            permission=permission,
            reason=reason,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_id=resource_id,
        )

        self._log_access(result)
        return result

    async def can_read(
        self,
        ctx: "AuthorizationContext | None",
        item_id: str | None = None,
    ) -> bool:
        """Check if context can read knowledge items.

        Args:
            ctx: Authorization context
            item_id: Optional specific item ID

        Returns:
            True if read access is allowed
        """
        result = self._check_permission(ctx, self.PERMISSION_READ, item_id)
        return result.allowed

    async def can_write(
        self,
        ctx: "AuthorizationContext | None",
        item_id: str | None = None,
    ) -> bool:
        """Check if context can write/create knowledge items.

        Args:
            ctx: Authorization context
            item_id: Optional specific item ID

        Returns:
            True if write access is allowed
        """
        result = self._check_permission(ctx, self.PERMISSION_WRITE, item_id)
        return result.allowed

    async def can_update(
        self,
        ctx: "AuthorizationContext | None",
        item_id: str,
    ) -> bool:
        """Check if context can update a knowledge item.

        Args:
            ctx: Authorization context
            item_id: Item ID to update

        Returns:
            True if update access is allowed
        """
        result = self._check_permission(ctx, self.PERMISSION_UPDATE, item_id)
        return result.allowed

    async def can_delete(
        self,
        ctx: "AuthorizationContext | None",
        item_id: str,
    ) -> bool:
        """Check if context can delete a knowledge item.

        Args:
            ctx: Authorization context
            item_id: Item ID to delete

        Returns:
            True if delete access is allowed
        """
        result = self._check_permission(ctx, self.PERMISSION_DELETE, item_id)
        return result.allowed

    async def can_share(
        self,
        ctx: "AuthorizationContext | None",
        item_id: str,
    ) -> bool:
        """Check if context can share a knowledge item.

        Args:
            ctx: Authorization context
            item_id: Item ID to share

        Returns:
            True if share access is allowed
        """
        result = self._check_permission(ctx, self.PERMISSION_SHARE, item_id)
        return result.allowed

    async def can_admin(
        self,
        ctx: "AuthorizationContext | None",
    ) -> bool:
        """Check if context has admin access to Knowledge Mound.

        Args:
            ctx: Authorization context

        Returns:
            True if admin access is allowed
        """
        result = self._check_permission(ctx, self.PERMISSION_ADMIN)
        return result.allowed

    def filter_by_tenant(
        self,
        ctx: "AuthorizationContext | None",
        items: list[Any],
    ) -> list[Any]:
        """Filter items to only those accessible by the tenant.

        Enforces tenant isolation by filtering items to only those
        that belong to the user's tenant.

        Args:
            ctx: Authorization context
            items: List of items to filter

        Returns:
            Filtered list of items accessible by the tenant
        """
        if not self._enforce_tenant_isolation:
            return items

        if ctx is None:
            return items

        tenant_id = getattr(ctx, "tenant_id", None)
        if tenant_id is None:
            return items

        # Filter by tenant_id in item metadata
        filtered = []
        for item in items:
            item_tenant = None
            if hasattr(item, "tenant_id"):
                item_tenant = item.tenant_id
            elif hasattr(item, "metadata"):
                item_tenant = item.metadata.get("tenant_id")

            # Include if no tenant specified (global) or matches
            if item_tenant is None or item_tenant == tenant_id:
                filtered.append(item)

        return filtered

    def require_permission(
        self,
        permission: str,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Decorator to require a permission for a function.

        Usage:
            @ac.require_permission("knowledge:read")
            async def list_items(ctx, ...):
                ...

        Args:
            permission: Permission required

        Returns:
            Decorator function
        """

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            @wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Extract ctx from args or kwargs
                ctx = kwargs.get("ctx")
                if ctx is None and args:
                    # Try first arg
                    ctx = args[0] if hasattr(args[0], "user_id") else None

                result = self._check_permission(ctx, permission)
                if not result.allowed:
                    raise PermissionError(f"Permission denied: {permission} - {result.reason}")

                return await func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator

    def get_access_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent access log entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of access log dictionaries
        """
        return [r.to_dict() for r in self._access_log[-limit:]]


# Global instance
_knowledge_access_control: KnowledgeMoundAccessControl | None = None


def get_knowledge_access_control() -> KnowledgeMoundAccessControl:
    """Get or create the global Knowledge Mound access control instance."""
    global _knowledge_access_control
    if _knowledge_access_control is None:
        _knowledge_access_control = KnowledgeMoundAccessControl()
    return _knowledge_access_control


__all__ = [
    "KnowledgeAccessResult",
    "KnowledgeMoundAccessControl",
    "get_knowledge_access_control",
]
