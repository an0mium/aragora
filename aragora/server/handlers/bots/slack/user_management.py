"""
Slack User and Team Management.

This module handles user and workspace operations including:
- Workspace authorization checks
- User role lookups
- Organization mapping
"""

import logging
import os
from typing import Any

from aragora.audit.unified import audit_data
from aragora.server.handlers.base import HandlerResult, error_response

from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed

from .constants import (
    PERM_SLACK_ADMIN,
    RBAC_AVAILABLE,
    AuthorizationContext,
    check_permission,
    _validate_slack_team_id,
)

logger = logging.getLogger(__name__)


def get_org_from_team(team_id: str) -> str | None:
    """Get organization ID from Slack team/workspace ID.

    In production, this would query a database mapping Slack workspaces
    to Aragora organizations.

    Args:
        team_id: Slack team/workspace ID

    Returns:
        Organization ID if found, None otherwise
    """
    try:
        from aragora.storage.slack_workspace_store import get_slack_workspace_store

        store = get_slack_workspace_store()
        workspace_info = store.get_workspace(team_id)
        if workspace_info:
            return workspace_info.get("org_id")
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("Could not lookup org from team: %s", e)
    return None


def get_user_roles_from_slack(team_id: str, user_id: str) -> set[str]:
    """Get user roles based on Slack workspace membership.

    Maps Slack user to Aragora roles. In production, this would:
    1. Check if user is workspace admin -> grant 'admin' role
    2. Check workspace-specific role assignments
    3. Default to 'user' role for authorized workspace members

    Args:
        team_id: Slack workspace ID
        user_id: Slack user ID

    Returns:
        Set of role names
    """
    try:
        from aragora.storage.slack_workspace_store import get_slack_workspace_store

        store = get_slack_workspace_store()
        user_roles = store.get_user_roles(team_id, user_id)
        if user_roles:
            return set(user_roles)
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.debug("Could not lookup user roles: %s", e)

    # Default to basic user role for valid workspace members
    return {"user"}


def check_workspace_authorized(team_id: str) -> tuple[bool, str | None]:
    """Check if a Slack workspace is authorized to use Aragora.

    Args:
        team_id: Slack team/workspace ID

    Returns:
        Tuple of (is_authorized, error_message)
    """
    # Validate team_id format first
    valid, error = _validate_slack_team_id(team_id)
    if not valid:
        return False, error

    try:
        from aragora.storage.slack_workspace_store import get_slack_workspace_store

        store = get_slack_workspace_store()
        workspace = store.get_workspace(team_id)

        if not workspace:
            logger.warning("Unauthorized workspace attempted access: %s", team_id)
            return False, "Workspace not authorized"

        if workspace.get("revoked"):
            logger.warning("Revoked workspace attempted access: %s", team_id)
            return False, "Workspace access has been revoked"

        return True, None
    except (ImportError, AttributeError, RuntimeError) as e:
        # If workspace store is not available, allow access in dev mode
        env = os.environ.get("ARAGORA_ENV", "production").lower()
        if env in ("development", "dev", "local", "test"):
            logger.debug("Workspace store not available in dev mode: %s", e)
            return True, None
        logger.error("Workspace authorization check failed: %s", e)
        return False, "Workspace authorization check failed"


def build_auth_context_from_slack(
    team_id: str,
    user_id: str,
    channel_id: str | None = None,
) -> Any | None:
    """Build an AuthorizationContext from Slack request data.

    This creates an auth context based on Slack workspace and user information.
    The workspace (team_id) is used to lookup the associated organization.

    Args:
        team_id: Slack workspace/team ID
        user_id: Slack user ID
        channel_id: Optional Slack channel ID for workspace scoping

    Returns:
        AuthorizationContext if RBAC is available, None otherwise
    """
    if not RBAC_AVAILABLE or AuthorizationContext is None:
        return None

    try:
        # Lookup org_id from Slack team_id
        org_id = get_org_from_team(team_id)

        # Get user roles from Slack workspace mapping
        roles = get_user_roles_from_slack(team_id, user_id)

        return AuthorizationContext(
            user_id=f"slack:{user_id}",
            org_id=org_id,
            workspace_id=team_id,
            roles=roles,
        )
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        logger.debug("Could not build auth context from Slack data: %s", e)
        return None


def check_user_permission(
    team_id: str,
    user_id: str,
    permission_key: str,
    channel_id: str | None = None,
) -> HandlerResult | None:
    """Check if a Slack user has permission to perform an action.

    Args:
        team_id: Slack team/workspace ID
        user_id: Slack user ID
        permission_key: Permission to check (e.g., "slack.commands.execute")
        channel_id: Optional channel ID for workspace scoping

    Returns:
        Error response if permission denied, None if allowed
    """
    if not RBAC_AVAILABLE or check_permission is None:
        if rbac_fail_closed():
            return error_response("Service unavailable: access control module not loaded", 503)
        # RBAC not available - allow access (rely on signature verification)
        return None

    # First check if workspace is authorized
    authorized, error = check_workspace_authorized(team_id)
    if not authorized:
        return error_response(error or "Workspace not authorized", 403)

    # Build auth context
    context = build_auth_context_from_slack(team_id, user_id, channel_id)
    if context is None:
        # Can't build context - allow access in dev mode
        env = os.environ.get("ARAGORA_ENV", "production").lower()
        if env in ("development", "dev", "local", "test"):
            return None
        return error_response("Authorization context not available", 500)

    try:
        decision = check_permission(context, permission_key)
        if not decision.allowed:
            logger.warning(
                "Permission denied: %s for Slack user %s in workspace %s: %s",
                permission_key,
                user_id,
                team_id,
                decision.reason,
            )
            audit_data(
                user_id=f"slack:{user_id}",
                resource_type="slack_permission",
                resource_id=permission_key,
                action="denied",
                platform="slack",
                team_id=team_id,
                reason=decision.reason,
            )
            return error_response(f"Permission denied: {decision.reason}", 403)

        return None  # Permission granted
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
        logger.error("RBAC check failed for Slack request: %s", e)
        # Fail closed in production, open in dev
        env = os.environ.get("ARAGORA_ENV", "production").lower()
        if env in ("development", "dev", "local", "test"):
            return None
        return error_response("Authorization check failed", 500)


def check_user_permission_or_admin(
    team_id: str,
    user_id: str,
    permission_key: str,
    channel_id: str | None = None,
) -> HandlerResult | None:
    """Check if user has permission OR is admin.

    Admin users (slack.admin permission) bypass specific permission checks.

    Args:
        team_id: Slack team/workspace ID
        user_id: Slack user ID
        permission_key: Permission to check
        channel_id: Optional channel ID

    Returns:
        Error response if permission denied, None if allowed
    """
    if not RBAC_AVAILABLE or check_permission is None:
        if rbac_fail_closed():
            return error_response("Service unavailable: access control module not loaded", 503)
        return None

    context = build_auth_context_from_slack(team_id, user_id, channel_id)
    if context is None:
        return None

    try:
        # Check admin permission first
        admin_decision = check_permission(context, PERM_SLACK_ADMIN)
        if admin_decision.allowed:
            return None

        # Fall back to specific permission
        return check_user_permission(team_id, user_id, permission_key, channel_id)
    except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
        logger.debug("Admin check failed: %s", e)
        return check_user_permission(team_id, user_id, permission_key, channel_id)


__all__ = [
    "get_org_from_team",
    "get_user_roles_from_slack",
    "check_workspace_authorized",
    "build_auth_context_from_slack",
    "check_user_permission",
    "check_user_permission_or_admin",
]
