"""
Tenant Access Validation Utilities.

Provides secure validation of tenant/workspace access to prevent unauthorized
cross-tenant data access. All handlers extracting tenant_id from requests
should validate user access before proceeding.

Usage:
    from aragora.server.handlers.utils.tenant_validation import (
        validate_tenant_access,
        TenantAccessDeniedError,
        audit_cross_tenant_attempt,
    )

    # In a handler method:
    err = await validate_tenant_access(user, requested_tenant_id)
    if err:
        return err  # Returns 403 error response

    # Or with workspace_id:
    err = await validate_workspace_access(user, workspace_id)
    if err:
        return err
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)

# Audit logger for security events
_security_audit_logger = logging.getLogger("aragora.security.audit")


class TenantAccessDeniedError(Exception):
    """Raised when a user attempts to access a tenant they are not authorized for."""

    def __init__(
        self,
        user_id: str,
        requested_tenant_id: str,
        message: str = "Access denied to requested tenant",
    ):
        self.user_id = user_id
        self.requested_tenant_id = requested_tenant_id
        super().__init__(message)


def audit_cross_tenant_attempt(
    user_id: str,
    user_tenant_id: str | None,
    requested_tenant_id: str,
    endpoint: str = "unknown",
    ip_address: str | None = None,
    additional_context: dict[str, Any] | None = None,
) -> None:
    """
    Log a cross-tenant access attempt for security audit.

    This creates an audit log entry when a user attempts to access
    a tenant/workspace they don't have permission for.

    Args:
        user_id: ID of the user making the request
        user_tenant_id: The tenant the user belongs to
        requested_tenant_id: The tenant they attempted to access
        endpoint: The API endpoint being accessed
        ip_address: Client IP address if available
        additional_context: Any additional context for the audit log
    """
    audit_entry = {
        "event_type": "cross_tenant_access_attempt",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "user_tenant_id": user_tenant_id,
        "requested_tenant_id": requested_tenant_id,
        "endpoint": endpoint,
        "ip_address": ip_address,
        "severity": "warning",
        "additional_context": additional_context or {},
    }

    # Log to security audit logger
    _security_audit_logger.warning(
        "SECURITY: Cross-tenant access attempt detected | user=%s | user_tenant=%s | requested_tenant=%s | endpoint=%s",
        user_id,
        user_tenant_id,
        requested_tenant_id,
        endpoint,
    )

    # Also log the full structured entry for SIEM integration
    logger.warning("Cross-tenant access attempt: %s", audit_entry)


def _get_user_tenant_memberships(user: Any) -> set[str]:
    """
    Extract tenant/workspace memberships from user context.

    Handles various user context types and extracts all tenant IDs
    the user has access to.

    Args:
        user: UserAuthContext, AuthorizationContext, or similar user object

    Returns:
        Set of tenant IDs the user has access to
    """
    memberships: set[str] = set()

    # Direct tenant_id/org_id
    if hasattr(user, "tenant_id") and user.tenant_id:
        memberships.add(user.tenant_id)
    if hasattr(user, "org_id") and user.org_id:
        memberships.add(user.org_id)

    # Workspace ID (treat as equivalent to tenant for this context)
    if hasattr(user, "workspace_id") and user.workspace_id:
        memberships.add(user.workspace_id)

    # Workspace memberships array (if present)
    if hasattr(user, "workspace_memberships"):
        for ws in user.workspace_memberships or []:
            if isinstance(ws, dict):
                ws_id = ws.get("workspace_id") or ws.get("id")
                if ws_id:
                    memberships.add(ws_id)
            elif hasattr(ws, "workspace_id"):
                memberships.add(ws.workspace_id)
            elif hasattr(ws, "id"):
                memberships.add(ws.id)

    # Tenant memberships array (if present)
    if hasattr(user, "tenant_memberships"):
        for tm in user.tenant_memberships or []:
            if isinstance(tm, dict):
                t_id = tm.get("tenant_id") or tm.get("id")
                if t_id:
                    memberships.add(t_id)
            elif hasattr(tm, "tenant_id"):
                memberships.add(tm.tenant_id)
            elif hasattr(tm, "id"):
                memberships.add(tm.id)

    return memberships


def _is_admin_user(user: Any) -> bool:
    """
    Check if user has admin privileges that bypass tenant restrictions.

    Args:
        user: User context object

    Returns:
        True if user has system-level admin privileges
    """
    # Check is_admin flag
    if getattr(user, "is_admin", False):
        return True

    # Check is_superadmin flag
    if getattr(user, "is_superadmin", False):
        return True

    # Check for admin role
    roles = getattr(user, "roles", set()) or set()
    if isinstance(roles, (list, tuple)):
        roles = set(roles)

    if "admin" in roles or "super_admin" in roles or "superadmin" in roles:
        return True

    # Check single role attribute
    role = getattr(user, "role", None)
    if role in ("admin", "super_admin", "superadmin"):
        return True

    # Check for admin permission
    permissions = getattr(user, "permissions", set()) or set()
    if isinstance(permissions, (list, tuple)):
        permissions = set(permissions)

    if "admin" in permissions or "*" in permissions:
        return True

    return False


async def validate_tenant_access(
    user: Any,
    requested_tenant_id: str | None,
    endpoint: str = "unknown",
    ip_address: str | None = None,
    allow_none: bool = True,
) -> HandlerResult | None:
    """
    Validate that the authenticated user has access to the requested tenant.

    This function should be called whenever a tenant_id is extracted from
    request headers, query parameters, or body to ensure the user has
    permission to access that tenant's data.

    Args:
        user: Authenticated user context (UserAuthContext or AuthorizationContext)
        requested_tenant_id: The tenant ID the user is trying to access
        endpoint: The API endpoint for audit logging
        ip_address: Client IP for audit logging
        allow_none: If True, allows None tenant_id (defaults to user's tenant)

    Returns:
        None if access is allowed, or an error response dict if denied

    Usage:
        user, auth_err = self.require_auth_or_error(handler)
        if auth_err:
            return auth_err

        # Validate tenant access
        err = await validate_tenant_access(user, tenant_id, endpoint="/api/v2/integrations")
        if err:
            return err

        # Proceed with tenant-scoped operation
    """
    from aragora.server.handlers.utils.responses import error_response

    # If no tenant requested and allow_none, access is granted
    if requested_tenant_id is None:
        if allow_none:
            return None
        return error_response(
            "Tenant ID is required",
            400,
            code="TENANT_REQUIRED",
        )

    # Get user ID for audit logging
    user_id = getattr(user, "user_id", None) or getattr(user, "id", "unknown")

    # Check if user is admin (bypasses tenant restrictions)
    if _is_admin_user(user):
        logger.debug("Admin user %s granted access to tenant %s", user_id, requested_tenant_id)
        return None

    # Get user's tenant memberships
    memberships = _get_user_tenant_memberships(user)

    # Check if requested tenant is in user's memberships
    if requested_tenant_id in memberships:
        return None

    # Access denied - audit and return error
    user_tenant_id = (
        getattr(user, "tenant_id", None)
        or getattr(user, "org_id", None)
        or getattr(user, "workspace_id", None)
    )

    audit_cross_tenant_attempt(
        user_id=str(user_id),
        user_tenant_id=user_tenant_id,
        requested_tenant_id=requested_tenant_id,
        endpoint=endpoint,
        ip_address=ip_address,
    )

    return error_response(
        "Access denied to requested tenant",
        403,
        code="TENANT_ACCESS_DENIED",
    )


async def validate_workspace_access(
    user: Any,
    workspace_id: str | None,
    endpoint: str = "unknown",
    ip_address: str | None = None,
    allow_none: bool = True,
    allow_default: bool = True,
) -> HandlerResult | None:
    """
    Validate that the authenticated user has access to the requested workspace.

    This is an alias for validate_tenant_access with workspace-specific defaults.
    Workspaces are treated as equivalent to tenants for access control purposes.

    Args:
        user: Authenticated user context
        workspace_id: The workspace ID the user is trying to access
        endpoint: The API endpoint for audit logging
        ip_address: Client IP for audit logging
        allow_none: If True, allows None workspace_id
        allow_default: If True, allows "default" workspace for any authenticated user

    Returns:
        None if access is allowed, or an error response dict if denied
    """
    # "default" workspace is allowed for all authenticated users
    if allow_default and workspace_id == "default":
        return None

    return await validate_tenant_access(
        user=user,
        requested_tenant_id=workspace_id,
        endpoint=endpoint,
        ip_address=ip_address,
        allow_none=allow_none,
    )


def validate_tenant_access_sync(
    user: Any,
    requested_tenant_id: str | None,
    endpoint: str = "unknown",
    ip_address: str | None = None,
    allow_none: bool = True,
) -> HandlerResult | None:
    """
    Synchronous version of validate_tenant_access for non-async handlers.

    Args:
        user: Authenticated user context
        requested_tenant_id: The tenant ID the user is trying to access
        endpoint: The API endpoint for audit logging
        ip_address: Client IP for audit logging
        allow_none: If True, allows None tenant_id

    Returns:
        None if access is allowed, or an error response dict if denied
    """
    from aragora.server.handlers.utils.responses import error_response

    # If no tenant requested and allow_none, access is granted
    if requested_tenant_id is None:
        if allow_none:
            return None
        return error_response(
            "Tenant ID is required",
            400,
            code="TENANT_REQUIRED",
        )

    # Get user ID for audit logging
    user_id = getattr(user, "user_id", None) or getattr(user, "id", "unknown")

    # Check if user is admin (bypasses tenant restrictions)
    if _is_admin_user(user):
        logger.debug("Admin user %s granted access to tenant %s", user_id, requested_tenant_id)
        return None

    # Get user's tenant memberships
    memberships = _get_user_tenant_memberships(user)

    # Check if requested tenant is in user's memberships
    if requested_tenant_id in memberships:
        return None

    # Access denied - audit and return error
    user_tenant_id = (
        getattr(user, "tenant_id", None)
        or getattr(user, "org_id", None)
        or getattr(user, "workspace_id", None)
    )

    audit_cross_tenant_attempt(
        user_id=str(user_id),
        user_tenant_id=user_tenant_id,
        requested_tenant_id=requested_tenant_id,
        endpoint=endpoint,
        ip_address=ip_address,
    )

    return error_response(
        "Access denied to requested tenant",
        403,
        code="TENANT_ACCESS_DENIED",
    )


def validate_workspace_access_sync(
    user: Any,
    workspace_id: str | None,
    endpoint: str = "unknown",
    ip_address: str | None = None,
    allow_none: bool = True,
    allow_default: bool = True,
) -> HandlerResult | None:
    """
    Synchronous version of validate_workspace_access for non-async handlers.

    Args:
        user: Authenticated user context
        workspace_id: The workspace ID the user is trying to access
        endpoint: The API endpoint for audit logging
        ip_address: Client IP for audit logging
        allow_none: If True, allows None workspace_id
        allow_default: If True, allows "default" workspace for any authenticated user

    Returns:
        None if access is allowed, or an error response dict if denied
    """
    # "default" workspace is allowed for all authenticated users
    if allow_default and workspace_id == "default":
        return None

    return validate_tenant_access_sync(
        user=user,
        requested_tenant_id=workspace_id,
        endpoint=endpoint,
        ip_address=ip_address,
        allow_none=allow_none,
    )


def get_validated_tenant_id(
    user: Any,
    requested_tenant_id: str | None,
) -> str | None:
    """
    Get the validated tenant ID for a request, falling back to user's tenant.

    This is useful for handlers that want to default to the user's tenant
    when no explicit tenant is provided.

    Args:
        user: Authenticated user context
        requested_tenant_id: The explicitly requested tenant ID (may be None)

    Returns:
        The validated tenant ID to use (user's default if none requested)
    """
    if requested_tenant_id:
        return requested_tenant_id

    # Fall back to user's tenant
    return (
        getattr(user, "tenant_id", None)
        or getattr(user, "org_id", None)
        or getattr(user, "workspace_id", None)
    )
