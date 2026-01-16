"""
Multi-Tenancy Middleware.

Provides workspace isolation for multi-tenant SaaS operation.
Ensures users can only access resources within their workspace.

Usage:
    from aragora.server.middleware.tenancy import require_workspace, tenant_scoped

    @require_workspace
    def workspace_endpoint(self, handler, user: User, workspace: Workspace):
        return {"workspace_id": workspace.id}

    @tenant_scoped
    async def get_debates(workspace_id: str):
        # Automatically scoped to workspace
        return await db.query(Debate).filter_by(workspace_id=workspace_id)
"""

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from .auth_v2 import User, Workspace, get_current_user

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Plan Limits
# =============================================================================

PLAN_LIMITS = {
    "free": {
        "max_debates_per_month": 50,
        "max_agents": 2,
        "max_members": 1,
        "max_concurrent_debates": 1,
        "private_debates": False,
        "api_access": False,
        "priority_support": False,
    },
    "pro": {
        "max_debates_per_month": 500,
        "max_agents": 5,
        "max_members": 1,
        "max_concurrent_debates": 3,
        "private_debates": True,
        "api_access": True,
        "priority_support": False,
    },
    "team": {
        "max_debates_per_month": 2000,
        "max_agents": 10,
        "max_members": 10,
        "max_concurrent_debates": 10,
        "private_debates": True,
        "api_access": True,
        "priority_support": True,
    },
    "enterprise": {
        "max_debates_per_month": -1,  # Unlimited
        "max_agents": -1,  # Unlimited
        "max_members": -1,  # Unlimited
        "max_concurrent_debates": -1,  # Unlimited
        "private_debates": True,
        "api_access": True,
        "priority_support": True,
    },
}


def get_plan_limits(plan: str) -> Dict[str, Any]:
    """Get limits for a plan."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])


# =============================================================================
# Workspace Management
# =============================================================================


class WorkspaceManager:
    """
    Manages workspace operations and isolation.

    In production, this would connect to a database.
    For now, provides in-memory management.
    """

    def __init__(self, storage: Optional[Any] = None):
        self._storage = storage
        self._cache: Dict[str, Workspace] = {}

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        if workspace_id in self._cache:
            return self._cache[workspace_id]

        if self._storage:
            try:
                data = await self._storage.get_workspace(workspace_id)
                if data:
                    workspace = Workspace(**data)
                    self._cache[workspace_id] = workspace
                    return workspace
            except Exception as e:
                logger.error(f"Failed to get workspace: {e}")

        return None

    async def get_user_workspace(self, user: User) -> Optional[Workspace]:
        """Get the workspace for a user."""
        if user.workspace_id:
            return await self.get_workspace(user.workspace_id)

        # Create default workspace for user if none exists
        return await self.create_default_workspace(user)

    async def create_default_workspace(self, user: User) -> Workspace:
        """Create a default personal workspace for a user."""
        import uuid

        # Map plan limits to Workspace fields (PLAN_LIMITS uses different key names)
        limits = get_plan_limits(user.plan)
        workspace = Workspace(
            id=str(uuid.uuid4()),
            name=f"{user.email}'s Workspace",
            owner_id=user.id,
            plan=user.plan,
            max_debates=limits.get("max_debates_per_month", 50),
            max_agents=limits.get("max_agents", 2),
            max_members=limits.get("max_members", 1),
        )

        if self._storage:
            try:
                await self._storage.save_workspace(workspace)
            except Exception as e:
                logger.error(f"Failed to save workspace: {e}")

        self._cache[workspace.id] = workspace
        return workspace

    async def check_user_access(self, user: User, workspace_id: str) -> bool:
        """Check if user has access to a workspace."""
        workspace = await self.get_workspace(workspace_id)
        if not workspace:
            return False

        # Owner always has access
        if workspace.owner_id == user.id:
            return True

        # Check membership
        if user.id in workspace.member_ids:
            return True

        return False

    async def get_usage(self, workspace_id: str) -> Dict[str, Any]:
        """Get usage statistics for a workspace."""
        if self._storage:
            try:
                return await self._storage.get_workspace_usage(workspace_id)
            except Exception as e:
                logger.error(f"Failed to get usage: {e}")

        return {
            "debates_this_month": 0,
            "active_debates": 0,
            "total_tokens_used": 0,
        }

    async def check_limits(self, workspace: Workspace, action: str) -> tuple[bool, str]:
        """
        Check if a workspace can perform an action based on limits.

        Args:
            workspace: The workspace
            action: Action to check (e.g., "create_debate", "add_agent")

        Returns:
            (allowed, message) tuple
        """
        limits = get_plan_limits(workspace.plan)
        usage = await self.get_usage(workspace.id)

        if action == "create_debate":
            max_debates = limits["max_debates_per_month"]
            if max_debates != -1 and usage.get("debates_this_month", 0) >= max_debates:
                return False, f"Monthly debate limit ({max_debates}) reached"

            max_concurrent = limits["max_concurrent_debates"]
            if max_concurrent != -1 and usage.get("active_debates", 0) >= max_concurrent:
                return False, f"Concurrent debate limit ({max_concurrent}) reached"

        elif action == "add_agent":
            max_agents = limits["max_agents"]
            if max_agents != -1:
                # Would need to check current agent count
                pass

        elif action == "add_member":
            max_members = limits["max_members"]
            current = len(workspace.member_ids)
            if max_members != -1 and current >= max_members:
                return False, f"Member limit ({max_members}) reached"

        return True, ""


# =============================================================================
# Global Instance
# =============================================================================

_workspace_manager: Optional[WorkspaceManager] = None


def get_workspace_manager() -> WorkspaceManager:
    """Get the global workspace manager."""
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager()
    return _workspace_manager


# =============================================================================
# Decorators
# =============================================================================


def require_workspace(func: Callable) -> Callable:
    """
    Decorator that requires authenticated user with workspace access.

    Injects both 'user' and 'workspace' keyword arguments.

    Usage:
        @require_workspace
        def endpoint(self, handler, user: User, workspace: Workspace):
            return {"workspace_id": workspace.id}
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        from aragora.server.handlers.base import error_response

        # Extract handler
        handler = kwargs.get("handler")
        if handler is None:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            return error_response("No request handler", 500)

        # Authenticate user
        user = get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        # Get or create workspace
        manager = get_workspace_manager()
        workspace = await manager.get_user_workspace(user)

        if not workspace:
            return error_response("Workspace not found", 404)

        # Inject user and workspace
        kwargs["user"] = user
        kwargs["workspace"] = workspace

        # Call the function
        result = func(*args, **kwargs)
        if hasattr(result, "__await__"):
            result = await result
        return result

    return wrapper


def check_limit(action: str) -> Callable:
    """
    Decorator that checks workspace limits before executing.

    Args:
        action: The action to check (e.g., "create_debate")

    Usage:
        @require_workspace
        @check_limit("create_debate")
        def create_debate(self, handler, user, workspace):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from aragora.server.handlers.base import error_response

            workspace = kwargs.get("workspace")
            if not workspace:
                return error_response("Workspace required", 400)

            manager = get_workspace_manager()
            allowed, message = await manager.check_limits(workspace, action)

            if not allowed:
                return error_response(message, 403)

            result = func(*args, **kwargs)
            if hasattr(result, "__await__"):
                result = await result
            return result

        return wrapper

    return decorator


def tenant_scoped(func: Callable) -> Callable:
    """
    Decorator that ensures database queries are scoped to workspace.

    Validates workspace_id is provided and passes it to the wrapped function.

    Usage:
        @tenant_scoped
        async def get_debates(workspace_id: str):
            return await db.query(Debate).filter_by(workspace_id=workspace_id)
    """

    @wraps(func)
    async def wrapper(workspace_id: str, *args, **kwargs):
        if not workspace_id:
            raise ValueError("workspace_id is required for tenant-scoped queries")

        # Pass workspace_id as first positional argument only (not duplicated in kwargs)
        result = func(workspace_id, *args, **kwargs)
        if hasattr(result, "__await__"):
            result = await result
        return result

    return wrapper


# =============================================================================
# Utility Functions
# =============================================================================


def scope_query(query: Any, workspace_id: str) -> Any:
    """
    Add workspace scope to a database query.

    Works with SQLAlchemy-style queries.

    Args:
        query: The base query
        workspace_id: Workspace ID to scope to

    Returns:
        Scoped query
    """
    if hasattr(query, "filter"):
        return query.filter_by(workspace_id=workspace_id)
    elif hasattr(query, "where"):
        # SQLite-style
        return query.where("workspace_id = ?", (workspace_id,))
    return query


async def ensure_workspace_access(user: User, resource_workspace_id: str) -> bool:
    """
    Ensure user has access to a resource's workspace.

    Args:
        user: The authenticated user
        resource_workspace_id: Workspace ID of the resource

    Returns:
        True if user has access

    Raises:
        PermissionError if access denied
    """
    manager = get_workspace_manager()
    has_access = await manager.check_user_access(user, resource_workspace_id)

    if not has_access:
        raise PermissionError(f"Access denied to workspace {resource_workspace_id}")

    return True


__all__ = [
    # Limits
    "PLAN_LIMITS",
    "get_plan_limits",
    # Manager
    "WorkspaceManager",
    "get_workspace_manager",
    # Decorators
    "require_workspace",
    "check_limit",
    "tenant_scoped",
    # Utilities
    "scope_query",
    "ensure_workspace_access",
]
