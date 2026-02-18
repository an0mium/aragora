"""
Admin Handler - Main facade composing all admin endpoint mixins.

Provides the main AdminHandler class that combines:
- MetricsDashboardMixin: Platform metrics and statistics
- UserManagementMixin: User and organization management
- NomicAdminMixin: Nomic loop control

All endpoints require admin or owner role with MFA enabled (SOC 2 CC5-01).
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any
from collections.abc import Callable

# Pre-declare RBAC names for optional import fallback (avoids type: ignore on fallback definitions)
require_permission: Any

from aragora.audit.unified import audit_admin  # noqa: F401
from aragora.auth.lockout import get_lockout_tracker  # noqa: F401
from aragora.billing.jwt_auth import create_access_token  # noqa: F401
from aragora.billing.jwt_auth import extract_user_from_request
from aragora.server.middleware.mfa import enforce_admin_mfa_policy

from ..base import (
    SAFE_ID_PATTERN,
    HandlerResult,
    error_response,
    validate_path_segment,
)
from ..utils.rate_limit import RateLimiter, get_client_ip
from ..secure import (
    SecureHandler,
    UnauthorizedError,
    ForbiddenError,
)

# Import mixins
from .metrics_dashboard import MetricsDashboardMixin
from .users import UserManagementMixin, PERM_ADMIN_USERS_WRITE, PERM_ADMIN_IMPERSONATE
from .nomic_admin import NomicAdminMixin, PERM_ADMIN_NOMIC_WRITE, PERM_ADMIN_SYSTEM_WRITE

logger = logging.getLogger(__name__)

# Admin roles that can access admin endpoints
ADMIN_ROLES = {"admin", "owner"}

# Rate limiter for admin endpoints (10 requests per minute - sensitive operations)
_admin_limiter = RateLimiter(requests_per_minute=10)

# RBAC imports (optional - graceful degradation if not available)
try:
    from aragora.rbac import (
        AuthorizationContext,
        check_permission,
        PermissionDeniedError,
    )
    from aragora.rbac.decorators import require_permission

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed


if not RBAC_AVAILABLE:
    # Provide a no-op decorator fallback when RBAC is unavailable
    def require_permission(
        permission_key: str,
        resource_id_param: str | None = None,
        context_param: str = "context",
        checker: Any = None,
        on_denied: Any = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """No-op decorator when RBAC module is not available."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


# Metrics imports (optional)
try:
    from aragora.observability.metrics import record_rbac_check

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_rbac_check(*args: Any, **kwargs: Any) -> None:
        pass


def admin_secure_endpoint(
    permission: str | None = None,
    audit: bool = False,
    audit_action: str | None = None,
    resource_id_param: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Admin-specific secure endpoint decorator.

    Combines SecureHandler's @secure_endpoint with admin MFA enforcement.
    All admin endpoints require:
    1. Authentication (JWT)
    2. Admin or owner role
    3. MFA enabled (SOC 2 CC5-01)
    4. Optional RBAC permission check

    Args:
        permission: Required RBAC permission (e.g., "admin.users.impersonate")
        audit: Whether to log to audit trail
        audit_action: Custom action name for audit
        resource_id_param: Parameter containing resource ID

    Usage:
        @admin_secure_endpoint(permission="admin.users.impersonate", audit=True)
        async def _impersonate_user(self, request, auth_context, target_user_id):
            ...
    """
    from aragora.observability.immutable_log import get_audit_log
    from aragora.observability.metrics.security import (
        record_auth_attempt,
        record_blocked_request,
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(
            self: AdminHandler,
            request: Any,
            *args: Any,
            **kwargs: Any,
        ) -> HandlerResult:
            start_time = time.perf_counter()

            try:
                # 1. Get auth context
                auth_context = await self.get_auth_context(request, require_auth=True)
                record_auth_attempt("jwt", success=True)

                # 2. Check admin role
                user_store = self._get_user_store()
                if not user_store:
                    return error_response("Service unavailable", 503)

                user = user_store.get_user_by_id(auth_context.user_id)
                if not user or user.role not in ADMIN_ROLES:
                    logger.warning(f"Non-admin user {auth_context.user_id} attempted admin access")
                    record_blocked_request("admin_required", "user")
                    return error_response("Admin access required", 403)

                # 3. Enforce MFA for admin users (SOC 2 CC5-01)
                mfa_result = enforce_admin_mfa_policy(user, user_store)
                if mfa_result is not None:
                    reason = mfa_result.get("reason", "MFA required")
                    logger.warning(f"Admin user {auth_context.user_id} denied: {reason}")
                    record_blocked_request("mfa_required", "admin")
                    return error_response(
                        f"Administrative access requires MFA. {reason}",
                        403,
                        code="ADMIN_MFA_REQUIRED",
                    )

                # 4. Check RBAC permission if specified
                if permission:
                    resource_id = kwargs.get(resource_id_param) if resource_id_param else None
                    try:
                        self.check_permission(
                            auth_context, permission, str(resource_id) if resource_id else None
                        )
                    except ForbiddenError:
                        return error_response(f"Permission denied: {permission}", 403)

                # 5. Call the actual handler
                result = await func(self, request, auth_context, *args, **kwargs)

                # 6. Audit if requested
                if audit:
                    action_name = audit_action or func.__name__.lstrip("_")
                    resource_id = str(kwargs.get(resource_id_param, "system"))
                    await get_audit_log().append(
                        event_type=f"admin.{action_name}",
                        actor=auth_context.user_id,
                        actor_type="admin",
                        resource_type=self.RESOURCE_TYPE,
                        resource_id=resource_id,
                        action=action_name,
                        workspace_id=auth_context.workspace_id,
                        details={
                            "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
                            "permission_checked": permission,
                        },
                        ip_address=getattr(request, "remote", None),
                        user_agent=(
                            request.headers.get("User-Agent")
                            if hasattr(request, "headers")
                            else None
                        ),
                    )

                return result

            except (UnauthorizedError, ForbiddenError) as e:
                return self.handle_security_error(e, request)

        return wrapper

    return decorator


class AdminHandler(
    MetricsDashboardMixin,
    UserManagementMixin,
    NomicAdminMixin,
    SecureHandler,
):
    """
    Handler for admin endpoints.

    Extends SecureHandler with admin-specific MFA enforcement.
    All admin operations are audited to the immutable log.

    Composed of:
    - MetricsDashboardMixin: _get_stats, _get_system_metrics, _get_revenue_stats
    - UserManagementMixin: _list_organizations, _list_users, _impersonate_user,
                           _deactivate_user, _activate_user, _unlock_user
    - NomicAdminMixin: _get_nomic_status, _get_nomic_circuit_breakers,
                       _reset_nomic_phase, _pause_nomic, _resume_nomic,
                       _reset_nomic_circuit_breakers
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    # Resource type for audit logging
    RESOURCE_TYPE = "admin"

    ROUTES = [
        "/api/v1/admin/organizations",
        "/api/v1/admin/users",
        "/api/v1/admin/stats",
        "/api/v1/admin/system/metrics",
        "/api/v1/admin/impersonate",
        "/api/v1/admin/revenue",
        # Nomic admin endpoints
        "/api/v1/admin/nomic/status",
        "/api/v1/admin/nomic/circuit-breakers",
        "/api/v1/admin/nomic/reset",
        "/api/v1/admin/nomic/pause",
        "/api/v1/admin/nomic/resume",
        "/api/v1/admin/nomic/circuit-breakers/reset",
        # Circuit breaker management
        "/api/v1/admin/circuit-breakers",
        "/api/v1/admin/circuit-breakers/reset",
        # Security admin endpoints
        "/api/v1/admin/security/audit",
        "/api/v1/admin/security/compliance",
        "/api/v1/admin/security/scan",
        "/api/v1/admin/security/threats",
    ]

    @staticmethod
    def can_handle(path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/v1/admin")

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _require_admin(self, handler: Any) -> tuple[Any | None, HandlerResult | None]:
        """
        Verify the request is from an admin user with MFA enabled.

        SOC 2 Control: CC5-01 - Administrative access requires MFA.

        Returns:
            Tuple of (auth_context, error_response).
            If error_response is not None, return it immediately.
        """
        user_store = self._get_user_store()
        if not user_store:
            return None, error_response("Service unavailable", 503)

        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return None, error_response("Not authenticated", 401)

        # Check if user has admin role
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user or user.role not in ADMIN_ROLES:
            logger.warning(f"Non-admin user {auth_ctx.user_id} attempted admin access")
            return None, error_response("Admin access required", 403)

        # Enforce MFA for admin users (SOC 2 CC5-01)
        # Returns None if compliant, or dict with enforcement details if not
        mfa_policy_result = enforce_admin_mfa_policy(user, user_store)
        if mfa_policy_result is not None:
            reason = mfa_policy_result.get("reason", "MFA required")
            action = mfa_policy_result.get("action", "enable_mfa")
            logger.warning(f"Admin user {auth_ctx.user_id} denied: {reason} (action={action})")
            return None, error_response(
                f"Administrative access requires MFA. {reason}. "
                "Please enable MFA at /api/auth/mfa/setup",
                403,
                code="ADMIN_MFA_REQUIRED",
            )

        return auth_ctx, None

    def _check_rbac_permission(
        self, auth_ctx: Any, permission_key: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        """
        Check granular RBAC permission.

        For admin endpoints, this provides defense-in-depth beyond the basic admin
        role check. Admin/owner roles are allowed by default if RBAC isn't configured
        for the specific permission, but the check is logged for audit.

        Args:
            auth_ctx: Authentication context from _require_admin
            permission_key: Permission like "admin.users.impersonate"
            resource_id: Optional resource ID

        Returns:
            None if allowed, error response if denied
        """
        if not RBAC_AVAILABLE:
            if rbac_fail_closed():
                return error_response("Service unavailable: access control module not loaded", 503)
            # Fallback to role-based check (already done in _require_admin)
            return None

        try:
            # Build RBAC context from auth context
            user_store = self._get_user_store()
            if user_store is None:
                return None
            user = user_store.get_user_by_id(auth_ctx.user_id)
            roles = [user.role] if user else []
            user_role = user.role if user else None

            rbac_context = AuthorizationContext(
                user_id=auth_ctx.user_id,
                roles=set(roles),
                org_id=getattr(auth_ctx, "org_id", None),
            )

            decision = check_permission(rbac_context, permission_key, resource_id)
            if not decision.allowed:
                # For admin endpoints, admin/owner roles are allowed by default
                # if RBAC permission isn't explicitly configured. This provides
                # backward compatibility while still recording the check for audit.
                if user_role in ADMIN_ROLES:
                    logger.debug(
                        f"RBAC permission {permission_key} not explicitly granted, "
                        f"allowing admin {auth_ctx.user_id} by role fallback"
                    )
                    record_rbac_check(permission_key, granted=True)  # type: ignore[call-arg]
                    return None

                logger.warning(
                    f"RBAC permission denied: {permission_key} for user {auth_ctx.user_id}: {decision.reason}"
                )
                record_rbac_check(permission_key, granted=False)  # type: ignore[call-arg]
                return error_response("Permission denied", 403)
            record_rbac_check(permission_key, granted=True)  # type: ignore[call-arg]
        except PermissionDeniedError as e:
            logger.warning(
                f"RBAC permission denied: {permission_key} for user {auth_ctx.user_id}: {e}"
            )
            record_rbac_check(permission_key, granted=False)
            logger.warning("Handler error: %s", e)
            return error_response("Permission denied", 403)

        return None

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any, method: str = "GET"
    ) -> HandlerResult | None:
        """Route admin requests to appropriate methods."""
        # Rate limit check for admin endpoints
        client_ip = get_client_ip(handler)
        rate_key = client_ip
        test_name = os.environ.get("PYTEST_CURRENT_TEST")
        if test_name and client_ip not in _admin_limiter._buckets:
            rate_key = f"{client_ip}:{test_name}"
        if not _admin_limiter.is_allowed(rate_key):
            logger.warning(f"Rate limit exceeded for admin endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        # GET routes
        if method == "GET":
            if path == "/api/v1/admin/organizations":
                return self._list_organizations(handler, query_params)

            if path == "/api/v1/admin/users":
                return self._list_users(handler, query_params)

            if path == "/api/v1/admin/stats":
                return self._get_stats(handler)

            if path == "/api/v1/admin/system/metrics":
                return self._get_system_metrics(handler)

            if path == "/api/v1/admin/revenue":
                return self._get_revenue_stats(handler)

            # Nomic admin GET routes
            if path == "/api/v1/admin/nomic/status":
                return self._get_nomic_status(handler)

            if path == "/api/v1/admin/nomic/circuit-breakers":
                return self._get_nomic_circuit_breakers(handler)

        # POST routes
        if method == "POST":
            # POST /api/admin/impersonate/:user_id
            if path.startswith("/api/v1/admin/impersonate/"):
                user_id = path.split("/")[-1]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._impersonate_user(handler, user_id)

            # POST /api/admin/users/:user_id/deactivate
            if "/users/" in path and path.endswith("/deactivate"):
                parts = path.split("/")
                user_id = parts[-2]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._deactivate_user(handler, user_id)

            # POST /api/admin/users/:user_id/activate
            if "/users/" in path and path.endswith("/activate"):
                parts = path.split("/")
                user_id = parts[-2]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._activate_user(handler, user_id)

            # POST /api/admin/users/:user_id/unlock
            if "/users/" in path and path.endswith("/unlock"):
                parts = path.split("/")
                user_id = parts[-2]
                if not validate_path_segment(user_id, "user_id", SAFE_ID_PATTERN)[0]:
                    return error_response("Invalid user ID format", 400)
                return self._unlock_user(handler, user_id)

            # Nomic admin POST routes
            if path == "/api/v1/admin/nomic/reset":
                return self._reset_nomic_phase(handler)

            if path == "/api/v1/admin/nomic/pause":
                return self._pause_nomic(handler)

            if path == "/api/v1/admin/nomic/resume":
                return self._resume_nomic(handler)

            if path == "/api/v1/admin/nomic/circuit-breakers/reset":
                return self._reset_nomic_circuit_breakers(handler)

        return error_response("Method not allowed", 405)


__all__ = [
    "AdminHandler",
    "ADMIN_ROLES",
    "admin_secure_endpoint",
    "PERM_ADMIN_USERS_WRITE",
    "PERM_ADMIN_IMPERSONATE",
    "PERM_ADMIN_NOMIC_WRITE",
    "PERM_ADMIN_SYSTEM_WRITE",
]
