"""
RBAC Decorators - Easy permission enforcement for handlers.

Provides decorators for enforcing role-based access control on
handler methods and functions.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar, ParamSpec

from .checker import get_permission_checker, PermissionChecker
from .models import AuthorizationContext, AuthorizationDecision

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class PermissionDeniedError(Exception):
    """Raised when a permission check fails."""

    def __init__(
        self,
        message: str,
        decision: AuthorizationDecision | None = None,
    ) -> None:
        super().__init__(message)
        self.decision = decision
        self.permission_key = decision.permission_key if decision else None
        self.resource_id = decision.resource_id if decision else None


class RoleRequiredError(Exception):
    """Raised when a required role is missing."""

    def __init__(
        self,
        message: str,
        required_roles: set[str],
        actual_roles: set[str],
    ) -> None:
        super().__init__(message)
        self.required_roles = required_roles
        self.actual_roles = actual_roles


def _get_context_from_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    context_param: str,
) -> AuthorizationContext | None:
    """Extract authorization context from function arguments."""
    # Check kwargs first
    if context_param in kwargs:
        ctx = kwargs[context_param]
        if isinstance(ctx, AuthorizationContext):
            return ctx

    # Check if first arg is context
    if args and isinstance(args[0], AuthorizationContext):
        return args[0]

    # Check if it's a method with self/cls
    if len(args) >= 2 and isinstance(args[1], AuthorizationContext):
        return args[1]

    # Try to find context in kwargs by type
    for value in kwargs.values():
        if isinstance(value, AuthorizationContext):
            return value

    return None


def require_permission(
    permission_key: str,
    resource_id_param: str | None = None,
    context_param: str = "context",
    checker: PermissionChecker | None = None,
    on_denied: Callable[[AuthorizationDecision], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to require a specific permission.

    Args:
        permission_key: Permission required (e.g., "debates.create")
        resource_id_param: Optional parameter name containing resource ID
        context_param: Parameter name for AuthorizationContext
        checker: Custom permission checker (uses global if None)
        on_denied: Optional callback when permission is denied

    Usage:
        @require_permission("debates.create")
        async def create_debate(context: AuthorizationContext, ...):
            ...

        @require_permission("debates.update", resource_id_param="debate_id")
        async def update_debate(context: AuthorizationContext, debate_id: str, ...):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if context is None:
                raise PermissionDeniedError(
                    f"No AuthorizationContext found for permission check: {permission_key}"
                )

            # Get resource ID if specified
            resource_id: str | None = None
            if resource_id_param:
                raw_resource_id = kwargs.get(resource_id_param)
                if raw_resource_id is not None:
                    resource_id = str(raw_resource_id)
                else:
                    # Try positional args - this is fragile but necessary
                    import inspect

                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if resource_id_param in params:
                        idx = params.index(resource_id_param)
                        if idx < len(args):
                            resource_id = str(args[idx])

            # Check permission
            perm_checker = checker or get_permission_checker()
            decision = perm_checker.check_permission(context, permission_key, resource_id)

            if not decision.allowed:
                if on_denied:
                    on_denied(decision)
                raise PermissionDeniedError(decision.reason, decision)

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if context is None:
                raise PermissionDeniedError(
                    f"No AuthorizationContext found for permission check: {permission_key}"
                )

            # Get resource ID if specified
            resource_id: str | None = None
            if resource_id_param:
                raw_resource_id = kwargs.get(resource_id_param)
                if raw_resource_id is not None:
                    resource_id = str(raw_resource_id)

            # Check permission
            perm_checker = checker or get_permission_checker()
            decision = perm_checker.check_permission(context, permission_key, resource_id)

            if not decision.allowed:
                if on_denied:
                    on_denied(decision)
                raise PermissionDeniedError(decision.reason, decision)

            return await func(*args, **kwargs)  # type: ignore

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def require_role(
    *role_names: str,
    require_all: bool = False,
    context_param: str = "context",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to require specific role(s).

    Args:
        *role_names: Role names required
        require_all: If True, all roles required. If False, any role is sufficient.
        context_param: Parameter name for AuthorizationContext

    Usage:
        @require_role("admin")
        async def admin_action(context: AuthorizationContext, ...):
            ...

        @require_role("owner", "admin")  # Either role works
        async def privileged_action(context: AuthorizationContext, ...):
            ...

        @require_role("admin", "billing_manager", require_all=True)  # Both required
        async def billing_admin_action(context: AuthorizationContext, ...):
            ...
    """
    required_roles = set(role_names)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if not context:
                raise RoleRequiredError(
                    "No AuthorizationContext found for role check",
                    required_roles,
                    set(),
                )

            if require_all:
                if not required_roles <= context.roles:
                    missing = required_roles - context.roles
                    raise RoleRequiredError(
                        f"Missing required roles: {missing}",
                        required_roles,
                        context.roles,
                    )
            else:
                if not (required_roles & context.roles):
                    raise RoleRequiredError(
                        f"Requires one of roles: {required_roles}",
                        required_roles,
                        context.roles,
                    )

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if not context:
                raise RoleRequiredError(
                    "No AuthorizationContext found for role check",
                    required_roles,
                    set(),
                )

            if require_all:
                if not required_roles <= context.roles:
                    missing = required_roles - context.roles
                    raise RoleRequiredError(
                        f"Missing required roles: {missing}",
                        required_roles,
                        context.roles,
                    )
            else:
                if not (required_roles & context.roles):
                    raise RoleRequiredError(
                        f"Requires one of roles: {required_roles}",
                        required_roles,
                        context.roles,
                    )

            return await func(*args, **kwargs)  # type: ignore

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def require_owner(
    context_param: str = "context",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to require owner role."""
    return require_role("owner", context_param=context_param)


def require_admin(
    context_param: str = "context",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to require admin or owner role."""
    return require_role("owner", "admin", context_param=context_param)


def require_org_access(
    org_id_param: str = "org_id",
    context_param: str = "context",
    allow_none: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to require access to a specific organization.

    Args:
        org_id_param: Parameter name containing organization ID
        context_param: Parameter name for AuthorizationContext
        allow_none: If True, allow access when org_id is None

    Usage:
        @require_org_access()
        async def org_action(context: AuthorizationContext, org_id: str, ...):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if not context:
                raise PermissionDeniedError("No AuthorizationContext found")

            org_id = kwargs.get(org_id_param)

            if org_id is None and allow_none:
                return func(*args, **kwargs)

            if org_id is None:
                raise PermissionDeniedError("Organization ID required but not provided")

            # Check if user belongs to the organization
            if context.org_id != org_id:
                # Allow if user is a platform admin (no org_id but has admin role)
                if context.org_id is None and "owner" in context.roles:
                    return func(*args, **kwargs)
                raise PermissionDeniedError(f"User does not have access to organization {org_id}")

            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if not context:
                raise PermissionDeniedError("No AuthorizationContext found")

            org_id = kwargs.get(org_id_param)

            if org_id is None and allow_none:
                return await func(*args, **kwargs)  # type: ignore

            if org_id is None:
                raise PermissionDeniedError("Organization ID required but not provided")

            if context.org_id != org_id:
                if context.org_id is None and "owner" in context.roles:
                    return await func(*args, **kwargs)  # type: ignore
                raise PermissionDeniedError(f"User does not have access to organization {org_id}")

            return await func(*args, **kwargs)  # type: ignore

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def require_self_or_admin(
    user_id_param: str = "user_id",
    context_param: str = "context",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to require either the user is acting on themselves or is an admin.

    Useful for endpoints where users can modify their own data, or admins can
    modify any user's data.

    Usage:
        @require_self_or_admin()
        async def update_user(context: AuthorizationContext, user_id: str, ...):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if not context:
                raise PermissionDeniedError("No AuthorizationContext found")

            target_user_id = kwargs.get(user_id_param)

            # Allow if acting on self
            if target_user_id == context.user_id:
                return func(*args, **kwargs)

            # Allow if admin or owner
            if context.has_any_role("owner", "admin"):
                return func(*args, **kwargs)

            raise PermissionDeniedError("Can only modify own data or requires admin role")

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            context = _get_context_from_args(args, kwargs, context_param)
            if not context:
                raise PermissionDeniedError("No AuthorizationContext found")

            target_user_id = kwargs.get(user_id_param)

            if target_user_id == context.user_id:
                return await func(*args, **kwargs)  # type: ignore

            if context.has_any_role("owner", "admin"):
                return await func(*args, **kwargs)  # type: ignore

            raise PermissionDeniedError("Can only modify own data or requires admin role")

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def with_permission_context(
    user_id_func: Callable[..., str],
    org_id_func: Callable[..., str | None] | None = None,
    roles_func: Callable[..., set[str]] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator that builds AuthorizationContext from request data.

    Useful when the handler doesn't receive a context directly but needs
    to build one from request headers or session data.

    Args:
        user_id_func: Function to extract user ID from args/kwargs
        org_id_func: Optional function to extract org ID
        roles_func: Optional function to extract roles

    Usage:
        @with_permission_context(
            user_id_func=lambda request: request.user_id,
            org_id_func=lambda request: request.org_id,
            roles_func=lambda request: request.roles,
        )
        @require_permission("debates.create")
        async def create_debate(context: AuthorizationContext, request: Request):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Build context
            user_id = user_id_func(*args, **kwargs)
            org_id = org_id_func(*args, **kwargs) if org_id_func else None
            roles = roles_func(*args, **kwargs) if roles_func else set()

            context = AuthorizationContext(
                user_id=user_id,
                org_id=org_id,
                roles=roles,
            )

            # Add context to kwargs
            kwargs["context"] = context

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
