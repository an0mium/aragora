"""
Admin User and Organization Management Endpoints.

Provides administrative endpoints for managing users and organizations.
All endpoints require admin or owner role with MFA enabled.

Endpoints:
- GET /api/v1/admin/organizations - List all organizations
- GET /api/v1/admin/users - List all users
- POST /api/v1/admin/impersonate/:user_id - Create impersonation token
- POST /api/v1/admin/users/:user_id/deactivate - Deactivate a user
- POST /api/v1/admin/users/:user_id/activate - Activate a user
- POST /api/v1/admin/users/:user_id/unlock - Unlock a locked user account
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from aragora.auth.lockout import get_lockout_tracker
from aragora.audit.unified import audit_admin
from aragora.events.handler_events import emit_handler_event, UPDATED
from aragora.rbac.decorators import require_permission

from ..base import (
    SAFE_ID_PATTERN,
    HandlerResult,
    error_response,
    get_string_param,
    handle_errors,
    json_response,
    log_request,
    validate_path_segment,
)
from ..openapi_decorator import api_endpoint
from ..utils.sanitization import sanitize_user_response

if TYPE_CHECKING:
    from aragora.auth.context import AuthorizationContext
    from aragora.auth.store import UserStore

logger = logging.getLogger(__name__)

# RBAC Permission Constants for User Operations
PERM_ADMIN_USERS_WRITE = "admin:users:write"  # Deactivate/activate users
PERM_ADMIN_IMPERSONATE = "admin:impersonate"  # Create impersonation tokens


class UserManagementMixin:
    """
    Mixin providing user and organization management endpoints for admin.

    This mixin requires the following attributes from the base class:
    - ctx: dict[str, Any] - Server context
    - _get_user_store() -> UserStore | None
    - _require_admin(handler) -> tuple[AuthContext | None, HandlerResult | None]
    - _check_rbac_permission(auth_ctx, permission, resource_id=None) -> HandlerResult | None
    """

    # Type stubs for methods expected from host class (BaseHandler)
    ctx: dict[str, Any]
    _require_admin: Callable[[Any], tuple["AuthorizationContext | None", HandlerResult | None]]
    _check_rbac_permission: Callable[..., HandlerResult | None]
    _get_user_store: Callable[[], "UserStore | None"]

    @api_endpoint(
        method="GET",
        path="/api/v1/admin/organizations",
        summary="List all organizations",
        tags=["Admin"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            {"name": "tier", "in": "query", "schema": {"type": "string"}},
        ],
        responses={
            "200": {
                "description": "Paginated list of organizations",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "organizations": {"type": "array", "items": {"type": "object"}},
                                "total": {"type": "integer"},
                                "limit": {"type": "integer"},
                                "offset": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @require_permission("admin:organizations:list")
    @handle_errors("list organizations")
    def _list_organizations(self, handler: Any, query_params: dict[str, Any]) -> HandlerResult:
        """List all organizations with pagination.
        Requires admin:organizations:list permission.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission
        perm_err = self._check_rbac_permission(auth_ctx, "admin.organizations.list")
        if perm_err:
            return perm_err

        user_store = self._get_user_store()

        # Parse pagination params
        limit = min(int(get_string_param(query_params, "limit", "50")), 100)
        offset = int(get_string_param(query_params, "offset", "0"))
        tier_filter = get_string_param(query_params, "tier", None)

        organizations, total = user_store.list_all_organizations(
            limit=limit,
            offset=offset,
            tier_filter=tier_filter,
        )

        return json_response(
            {
                "organizations": [org.to_dict() for org in organizations],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @api_endpoint(
        method="GET",
        path="/api/v1/admin/users",
        summary="List all users",
        tags=["Admin"],
        parameters=[
            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 50}},
            {"name": "offset", "in": "query", "schema": {"type": "integer", "default": 0}},
            {"name": "org_id", "in": "query", "schema": {"type": "string"}},
            {"name": "role", "in": "query", "schema": {"type": "string"}},
            {
                "name": "active_only",
                "in": "query",
                "schema": {"type": "string", "default": "false"},
            },
        ],
        responses={
            "200": {
                "description": "Paginated list of users",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "users": {"type": "array", "items": {"type": "object"}},
                                "total": {"type": "integer"},
                                "limit": {"type": "integer"},
                                "offset": {"type": "integer"},
                            },
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @handle_errors("list users")
    def _list_users(self, handler: Any, query_params: dict[str, Any]) -> HandlerResult:
        """List all users with pagination and filtering.
        Requires admin:users:list permission.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission
        perm_err = self._check_rbac_permission(auth_ctx, "admin.users.list")
        if perm_err:
            return perm_err

        user_store = self._get_user_store()

        # Parse params
        limit = min(int(get_string_param(query_params, "limit", "50")), 100)
        offset = int(get_string_param(query_params, "offset", "0"))
        org_id = get_string_param(query_params, "org_id", None)
        role = get_string_param(query_params, "role", None)
        active_only = get_string_param(query_params, "active_only", "false").lower() == "true"

        users, total = user_store.list_all_users(
            limit=limit,
            offset=offset,
            org_id_filter=org_id,
            role_filter=role,
            active_only=active_only,
        )

        # Convert users to safe dict (exclude password hashes and sensitive fields)
        user_dicts = [sanitize_user_response(user.to_dict()) for user in users]

        return json_response(
            {
                "users": user_dicts,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/impersonate/{user_id}",
        summary="Create impersonation token for a user",
        tags=["Admin"],
        parameters=[
            {"name": "user_id", "in": "path", "required": True, "schema": {"type": "string"}},
        ],
        responses={
            "200": {
                "description": "Impersonation token created",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "token": {"type": "string"},
                                "expires_in": {"type": "integer"},
                                "target_user": {"type": "object"},
                                "warning": {"type": "string"},
                            },
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
            "404": {"description": "User not found"},
        },
    )
    @handle_errors("impersonate user")
    @log_request("admin impersonate")
    def _impersonate_user(self, handler: Any, target_user_id: str) -> HandlerResult:
        """
        Create an impersonation token for a user.

        This allows admins to view the system as a specific user for support.
        The token is short-lived (1 hour) and logged for audit.

        Requires permission: admin:impersonate
        """
        # Validate user_id path parameter
        if not validate_path_segment(target_user_id, "target_user_id", SAFE_ID_PATTERN)[0]:
            return error_response("Invalid target user ID format", 400)

        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check granular RBAC permission for impersonation (CRITICAL: admin:impersonate)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_IMPERSONATE, target_user_id)
        if perm_err:
            return perm_err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Create short-lived impersonation token (1 hour)
        # Note: impersonation metadata is logged below since JWT doesn't support custom claims
        from aragora.server.handlers.admin import handler as admin_handler

        impersonation_token = admin_handler.create_access_token(
            user_id=target_user_id,
            email=target_user.email,
            org_id=target_user.org_id,
            role=target_user.role,
            expiry_hours=1,
        )

        # Log the impersonation for audit
        logger.info(f"Admin {auth_ctx.user_id} impersonating user {target_user_id}")

        # Record in audit log if available
        try:
            user_store.record_audit_event(
                user_id=auth_ctx.user_id,
                org_id=None,
                event_type="admin_impersonate",
                action="impersonate_user",
                resource_type="user",
                resource_id=target_user_id,
                ip_address=getattr(handler, "client_address", ("unknown",))[0],
                details={"target_email": target_user.email},
            )
        except Exception as e:
            logger.warning(f"Failed to record audit event: {e}")

        return json_response(
            {
                "token": impersonation_token,
                "expires_in": 3600,
                "target_user": {
                    "id": target_user.id,
                    "email": target_user.email,
                    "name": target_user.name,
                    "role": target_user.role,
                },
                "warning": "This token grants full access as the target user. Use responsibly.",
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/users/{user_id}/deactivate",
        summary="Deactivate a user account",
        tags=["Admin"],
        parameters=[
            {"name": "user_id", "in": "path", "required": True, "schema": {"type": "string"}},
        ],
        responses={
            "200": {"description": "User deactivated successfully"},
            "400": {"description": "Cannot deactivate yourself"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
            "404": {"description": "User not found"},
        },
    )
    @handle_errors("deactivate user")
    @log_request("admin deactivate user")
    def _deactivate_user(self, handler: Any, target_user_id: str) -> HandlerResult:
        """Deactivate a user account.

        Requires permission: admin:users:write
        """
        # Validate user_id path parameter
        if not validate_path_segment(target_user_id, "target_user_id", SAFE_ID_PATTERN)[0]:
            return error_response("Invalid target user ID format", 400)

        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check granular RBAC permission (CRITICAL: admin:users:write)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_USERS_WRITE, target_user_id)
        if perm_err:
            return perm_err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Prevent deactivating yourself
        if target_user_id == auth_ctx.user_id:
            return error_response("Cannot deactivate yourself", 400)

        # Deactivate the user
        user_store.update_user(target_user_id, is_active=False)

        logger.info(f"Admin {auth_ctx.user_id} deactivated user {target_user_id}")
        audit_admin(
            admin_id=auth_ctx.user_id,
            action="deactivate_user",
            target_type="user",
            target_id=target_user_id,
            target_email=target_user.email,
        )

        emit_handler_event(
            "admin", UPDATED, {"action": "deactivate_user", "target_user_id": target_user_id}
        )
        return json_response(
            {
                "success": True,
                "user_id": target_user_id,
                "is_active": False,
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/users/{user_id}/activate",
        summary="Activate a user account",
        tags=["Admin"],
        parameters=[
            {"name": "user_id", "in": "path", "required": True, "schema": {"type": "string"}},
        ],
        responses={
            "200": {"description": "User activated successfully"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
            "404": {"description": "User not found"},
        },
    )
    @handle_errors("activate user")
    @log_request("admin activate user")
    def _activate_user(self, handler: Any, target_user_id: str) -> HandlerResult:
        """Activate a user account.

        Requires permission: admin:users:write
        """
        # Validate user_id path parameter
        if not validate_path_segment(target_user_id, "target_user_id", SAFE_ID_PATTERN)[0]:
            return error_response("Invalid target user ID format", 400)

        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check granular RBAC permission (CRITICAL: admin:users:write)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_USERS_WRITE, target_user_id)
        if perm_err:
            return perm_err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Activate the user
        user_store.update_user(target_user_id, is_active=True)

        logger.info(f"Admin {auth_ctx.user_id} activated user {target_user_id}")
        audit_admin(
            admin_id=auth_ctx.user_id,
            action="activate_user",
            target_type="user",
            target_id=target_user_id,
            target_email=target_user.email,
        )

        return json_response(
            {
                "success": True,
                "user_id": target_user_id,
                "is_active": True,
            }
        )

    @api_endpoint(
        method="POST",
        path="/api/v1/admin/users/{user_id}/unlock",
        summary="Unlock a locked user account",
        tags=["Admin"],
        parameters=[
            {"name": "user_id", "in": "path", "required": True, "schema": {"type": "string"}},
        ],
        responses={
            "200": {"description": "User account unlocked successfully"},
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
            "404": {"description": "User not found"},
        },
    )
    @handle_errors("unlock user")
    @log_request("admin unlock user")
    def _unlock_user(self, handler: Any, target_user_id: str) -> HandlerResult:
        """
        Unlock a user account that has been locked due to failed login attempts.

        This clears both the in-memory/Redis lockout tracker and the database
        lockout state. Use this to help users who have been locked out.

        Endpoint: POST /api/admin/users/:user_id/unlock

        Requires permission: admin:users:write
        """
        # Validate user_id path parameter
        if not validate_path_segment(target_user_id, "target_user_id", SAFE_ID_PATTERN)[0]:
            return error_response("Invalid target user ID format", 400)

        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check granular RBAC permission (CRITICAL: admin:users:write)
        perm_err = self._check_rbac_permission(auth_ctx, PERM_ADMIN_USERS_WRITE, target_user_id)
        if perm_err:
            return perm_err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        email = target_user.email

        # Get lockout info before clearing
        lockout_tracker = get_lockout_tracker()
        lockout_info = lockout_tracker.get_info(email=email)

        # Clear lockout tracker (in-memory/Redis)
        lockout_cleared = lockout_tracker.admin_unlock(
            email=email,
            user_id=target_user_id,
        )

        # Clear database lockout state if user store supports it
        db_cleared = False
        if hasattr(user_store, "reset_failed_login_attempts"):
            db_cleared = user_store.reset_failed_login_attempts(email)

        logger.info(
            f"Admin {auth_ctx.user_id} unlocked user {target_user_id} "
            f"(email={email}, tracker_cleared={lockout_cleared}, db_cleared={db_cleared})"
        )

        # Log audit event
        try:
            if hasattr(user_store, "log_audit_event"):
                user_store.log_audit_event(
                    action="admin_unlock_user",
                    resource_type="user",
                    resource_id=target_user_id,
                    user_id=auth_ctx.user_id,
                    metadata={
                        "target_email": email,
                        "lockout_info": lockout_info,
                    },
                    ip_address=getattr(handler, "client_address", ("unknown",))[0],
                )
        except Exception as e:
            logger.warning(f"Failed to record audit event: {e}")

        return json_response(
            {
                "success": True,
                "user_id": target_user_id,
                "email": email,
                "lockout_cleared": lockout_cleared or db_cleared,
                "previous_lockout_info": lockout_info,
                "message": f"Account lockout cleared for {email}",
            }
        )


__all__ = ["UserManagementMixin", "PERM_ADMIN_USERS_WRITE", "PERM_ADMIN_IMPERSONATE"]
