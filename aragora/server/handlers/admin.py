"""
Admin API Handlers.

Provides administrative endpoints for system-wide management.
All endpoints require admin or owner role.

Endpoints:
- GET /api/admin/organizations - List all organizations
- GET /api/admin/users - List all users
- GET /api/admin/stats - Get system-wide statistics
- GET /api/admin/system/metrics - Get aggregated system metrics
- POST /api/admin/impersonate/:user_id - Create impersonation token
- POST /api/admin/users/:user_id/deactivate - Deactivate a user
- POST /api/admin/users/:user_id/activate - Activate a user
- POST /api/admin/users/:user_id/unlock - Unlock a locked user account
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
    get_int_param,
    get_string_param,
    get_bool_param,
    validate_path_segment,
    SAFE_ID_PATTERN,
)
from aragora.billing.jwt_auth import extract_user_from_request, create_access_token
from aragora.auth.lockout import get_lockout_tracker

logger = logging.getLogger(__name__)

# Admin roles that can access admin endpoints
ADMIN_ROLES = {"admin", "owner"}


class AdminHandler(BaseHandler):
    """Handler for admin endpoints."""

    ROUTES = [
        "/api/admin/organizations",
        "/api/admin/users",
        "/api/admin/stats",
        "/api/admin/system/metrics",
        "/api/admin/impersonate",
        "/api/admin/revenue",
    ]

    @staticmethod
    def can_handle(path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/admin")

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _require_admin(self, handler) -> tuple[Optional[Any], Optional[HandlerResult]]:
        """
        Verify the request is from an admin user.

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
            logger.warning(
                f"Non-admin user {auth_ctx.user_id} attempted admin access"
            )
            return None, error_response("Admin access required", 403)

        return auth_ctx, None

    def handle(
        self, path: str, query_params: dict, handler, method: str = "GET"
    ) -> Optional[HandlerResult]:
        """Route admin requests to appropriate methods."""
        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

        # GET routes
        if method == "GET":
            if path == "/api/admin/organizations":
                return self._list_organizations(handler, query_params)

            if path == "/api/admin/users":
                return self._list_users(handler, query_params)

            if path == "/api/admin/stats":
                return self._get_stats(handler)

            if path == "/api/admin/system/metrics":
                return self._get_system_metrics(handler)

            if path == "/api/admin/revenue":
                return self._get_revenue_stats(handler)

        # POST routes
        if method == "POST":
            # POST /api/admin/impersonate/:user_id
            if path.startswith("/api/admin/impersonate/"):
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

        return error_response("Method not allowed", 405)

    @handle_errors("list organizations")
    def _list_organizations(
        self, handler, query_params: dict
    ) -> HandlerResult:
        """List all organizations with pagination."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

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

        return json_response({
            "organizations": [org.to_dict() for org in organizations],
            "total": total,
            "limit": limit,
            "offset": offset,
        })

    @handle_errors("list users")
    def _list_users(self, handler, query_params: dict) -> HandlerResult:
        """List all users with pagination and filtering."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

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

        # Convert users to safe dict (exclude password hashes)
        user_dicts = []
        for user in users:
            user_dict = user.to_dict()
            # Remove sensitive fields
            user_dict.pop("password_hash", None)
            user_dict.pop("password_salt", None)
            user_dict.pop("api_key", None)
            user_dict.pop("api_key_hash", None)
            user_dicts.append(user_dict)

        return json_response({
            "users": user_dicts,
            "total": total,
            "limit": limit,
            "offset": offset,
        })

    @handle_errors("get admin stats")
    def _get_stats(self, handler) -> HandlerResult:
        """Get system-wide statistics."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()
        stats = user_store.get_admin_stats()

        return json_response({"stats": stats})

    @handle_errors("get system metrics")
    def _get_system_metrics(self, handler) -> HandlerResult:
        """Get aggregated system metrics from various sources."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        metrics: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Get user store stats
        user_store = self._get_user_store()
        if user_store:
            metrics["users"] = user_store.get_admin_stats()

        # Get debate storage stats if available
        debate_storage = self.ctx.get("debate_storage")
        if debate_storage and hasattr(debate_storage, "get_statistics"):
            try:
                metrics["debates"] = debate_storage.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get debate stats: {e}")
                metrics["debates"] = {"error": "unavailable"}

        # Get circuit breaker stats if available
        try:
            from aragora.resilience import get_circuit_breaker_status
            metrics["circuit_breakers"] = get_circuit_breaker_status()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to get circuit breaker stats: {e}")

        # Get cache stats if available
        try:
            from aragora.server.handlers.cache import get_cache_stats
            metrics["cache"] = get_cache_stats()
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")

        # Get rate limit stats if available
        try:
            from aragora.server.middleware.rate_limit import get_rate_limiter
            limiter = get_rate_limiter()
            if limiter and hasattr(limiter, "get_stats"):
                metrics["rate_limits"] = limiter.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get rate limit stats: {e}")

        return json_response({"metrics": metrics})

    @handle_errors("get revenue stats")
    def _get_revenue_stats(self, handler) -> HandlerResult:
        """Get revenue and billing statistics."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Get tier distribution from stats
        stats = user_store.get_admin_stats()
        tier_distribution = stats.get("tier_distribution", {})

        # Calculate monthly recurring revenue (MRR) based on tier counts
        from aragora.billing.models import TIER_LIMITS

        mrr_cents = 0
        tier_revenue = {}
        for tier_name, count in tier_distribution.items():
            tier_limits = TIER_LIMITS.get(tier_name)
            if tier_limits:
                tier_mrr = tier_limits.price_monthly_cents * count
                tier_revenue[tier_name] = {
                    "count": count,
                    "price_cents": tier_limits.price_monthly_cents,
                    "mrr_cents": tier_mrr,
                }
                mrr_cents += tier_mrr

        return json_response({
            "revenue": {
                "mrr_cents": mrr_cents,
                "mrr_dollars": mrr_cents / 100,
                "arr_dollars": (mrr_cents * 12) / 100,
                "tier_breakdown": tier_revenue,
                "total_organizations": stats.get("total_organizations", 0),
                "paying_organizations": sum(
                    count for tier, count in tier_distribution.items()
                    if tier != "free"
                ),
            }
        })

    @handle_errors("impersonate user")
    @log_request("admin impersonate")
    def _impersonate_user(self, handler, target_user_id: str) -> HandlerResult:
        """
        Create an impersonation token for a user.

        This allows admins to view the system as a specific user for support.
        The token is short-lived (1 hour) and logged for audit.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Create short-lived impersonation token (1 hour)
        # Note: impersonation metadata is logged below since JWT doesn't support custom claims
        impersonation_token = create_access_token(
            user_id=target_user_id,
            email=target_user.email,
            org_id=target_user.org_id,
            role=target_user.role,
            expiry_hours=1,
        )

        # Log the impersonation for audit
        logger.info(
            f"Admin {auth_ctx.user_id} impersonating user {target_user_id}"
        )

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

        return json_response({
            "token": impersonation_token,
            "expires_in": 3600,
            "target_user": {
                "id": target_user.id,
                "email": target_user.email,
                "name": target_user.name,
                "role": target_user.role,
            },
            "warning": "This token grants full access as the target user. Use responsibly.",
        })

    @handle_errors("deactivate user")
    @log_request("admin deactivate user")
    def _deactivate_user(self, handler, target_user_id: str) -> HandlerResult:
        """Deactivate a user account."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

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

        return json_response({
            "success": True,
            "user_id": target_user_id,
            "is_active": False,
        })

    @handle_errors("activate user")
    @log_request("admin activate user")
    def _activate_user(self, handler, target_user_id: str) -> HandlerResult:
        """Activate a user account."""
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        user_store = self._get_user_store()

        # Verify target user exists
        target_user = user_store.get_user_by_id(target_user_id)
        if not target_user:
            return error_response("User not found", 404)

        # Activate the user
        user_store.update_user(target_user_id, is_active=True)

        logger.info(f"Admin {auth_ctx.user_id} activated user {target_user_id}")

        return json_response({
            "success": True,
            "user_id": target_user_id,
            "is_active": True,
        })

    @handle_errors("unlock user")
    @log_request("admin unlock user")
    def _unlock_user(self, handler, target_user_id: str) -> HandlerResult:
        """
        Unlock a user account that has been locked due to failed login attempts.

        This clears both the in-memory/Redis lockout tracker and the database
        lockout state. Use this to help users who have been locked out.

        Endpoint: POST /api/admin/users/:user_id/unlock
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

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

        return json_response({
            "success": True,
            "user_id": target_user_id,
            "email": email,
            "lockout_cleared": lockout_cleared or db_cleared,
            "previous_lockout_info": lockout_info,
            "message": f"Account lockout cleared for {email}",
        })


__all__ = ["AdminHandler"]
