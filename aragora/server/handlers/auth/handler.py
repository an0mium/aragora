"""
User Authentication Handler.

Endpoints:
- POST /api/auth/register - Create a new user account
- POST /api/auth/login - Authenticate and get tokens
- POST /api/auth/logout - Invalidate current token (adds to blacklist)
- POST /api/auth/logout-all - Invalidate all tokens for user (logout all devices)
- POST /api/auth/refresh - Refresh access token (revokes old refresh token)
- POST /api/auth/revoke - Explicitly revoke a specific token
- GET /api/auth/me - Get current user information
- PUT /api/auth/me - Update current user information
- POST /api/auth/password - Change password
- POST /api/auth/api-key - Generate API key
- DELETE /api/auth/api-key - Revoke API key
- GET /api/auth/sessions - List active sessions for current user
- DELETE /api/auth/sessions/:id - Revoke a specific session
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

# Lockout tracker for brute-force protection

# Module-level imports for test mocking compatibility
from aragora.billing.jwt_auth import extract_user_from_request, validate_refresh_token

# RBAC imports
from aragora.rbac import AuthorizationContext, check_permission
from aragora.rbac.defaults import get_role_permissions

from .signup_handlers import (
    handle_accept_invite,
    handle_check_invite,
    handle_invite,
    handle_resend_verification,
    handle_setup_organization,
    handle_verify_email,
)
from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from ..utils.rate_limit import rate_limit
from ..secure import SecureHandler
from aragora.auth import lockout as _lockout_module
from aragora.auth.lockout import get_lockout_tracker  # noqa: F401
from aragora.server.versioning.compat import strip_version_prefix

# Import handlers from split modules
from .login import handle_register, handle_login
from .password import (
    handle_change_password,
    handle_forgot_password,
    handle_reset_password,
    send_password_reset_email,
)
from .api_keys import (
    handle_generate_api_key,
    handle_revoke_api_key,
    handle_list_api_keys,
    handle_revoke_api_key_prefix,
)
from .mfa import (
    handle_mfa_setup,
    handle_mfa_enable,
    handle_mfa_disable,
    handle_mfa_verify,
    handle_mfa_backup_codes,
)
from .sessions import handle_list_sessions, handle_revoke_session

# Unified audit logging
try:
    from aragora.audit.unified import (
        audit_login,
        audit_logout,
        audit_admin,
        audit_security,
    )

    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    audit_login = None
    audit_logout = None
    audit_admin = None
    audit_security = None

logger = logging.getLogger(__name__)
_ORIGINAL_LOCKOUT_TRACKER = _lockout_module.get_lockout_tracker


def _run_maybe_async(result: Any) -> Any:
    """Resolve a coroutine result if needed, otherwise return as-is."""
    if asyncio.iscoroutine(result):
        from aragora.server.http_utils import run_async

        return run_async(result)
    return result


class AuthHandler(SecureHandler):
    """Handler for user authentication endpoints.

    Extends SecureHandler for JWT-based authentication, RBAC permission
    enforcement, and security audit logging.
    """

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = server_context or ctx or {}

    # Resource type for audit logging
    RESOURCE_TYPE = "auth"

    # Routes use non-versioned paths (handler registry normalizes /api/v1/auth/* to /api/auth/*)
    ROUTES = [
        "/api/auth/register",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/auth/logout-all",
        "/api/auth/refresh",
        "/api/auth/revoke",
        "/api/auth/me",
        "/api/auth/password",
        "/api/auth/password/change",
        "/api/auth/password/forgot",
        "/api/auth/password/reset",
        "/api/auth/forgot-password",
        "/api/auth/reset-password",
        "/api/auth/profile",
        "/api/auth/api-key",
        "/api/auth/api-keys",
        "/api/auth/api-keys/*",
        "/api/auth/mfa/setup",
        "/api/auth/mfa/enable",
        "/api/auth/mfa/disable",
        "/api/auth/mfa/verify",
        "/api/auth/mfa/backup-codes",
        "/api/auth/mfa",
        "/api/auth/sessions",
        "/api/auth/sessions/*",  # For DELETE /api/auth/sessions/:id
        "/api/auth/verify-email",
        "/api/auth/verify-email/resend",
        "/api/auth/resend-verification",
        "/api/auth/setup-organization",
        "/api/auth/invite",
        "/api/auth/check-invite",
        "/api/auth/accept-invite",
        "/api/auth/health",
        # SDK aliases for API key management
        "/api/keys",
        "/api/keys/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = strip_version_prefix(path)
        if normalized in self.ROUTES:
            return True
        # Handle wildcard routes for session management
        if normalized.startswith("/api/auth/sessions/"):
            return True
        if normalized.startswith("/api/auth/api-keys/"):
            return True
        # SDK alias: /api/keys/* -> /api/auth/api-keys/*
        if normalized.startswith("/api/keys/"):
            return True
        return False

    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
        method: str | None = None,
    ) -> HandlerResult | None:
        """Route auth requests to appropriate methods.

        Paths are normalized (e.g., /api/v1/auth/me -> /api/auth/me).
        """
        # Normalize path to handle v1 routes
        path = strip_version_prefix(path)

        # Normalize SDK alias: /api/keys -> /api/auth/api-keys
        if path == "/api/keys":
            path = "/api/auth/api-keys"
        elif path.startswith("/api/keys/"):
            path = "/api/auth/api-keys/" + path[len("/api/keys/") :]

        # Determine HTTP method from handler
        method = method or (getattr(handler, "command", "GET") if handler else "GET")

        if path == "/api/auth/register" and method == "POST":
            return self._handle_register(handler)

        if path == "/api/auth/login" and method == "POST":
            return self._handle_login(handler)

        if path == "/api/auth/logout" and method == "POST":
            return self._handle_logout(handler)

        if path == "/api/auth/logout-all" and method == "POST":
            return self._handle_logout_all(handler)

        if path == "/api/auth/refresh" and method == "POST":
            return self._handle_refresh(handler)

        if path == "/api/auth/me":
            if method == "GET":
                return await self._handle_get_me(handler)
            elif method == "PUT":
                return self._handle_update_me(handler)
            elif method == "POST":
                return self._handle_update_me(handler)

        if path == "/api/auth/profile" and method == "POST":
            return self._handle_update_me(handler)

        if path == "/api/auth/password" and method == "POST":
            return self._handle_change_password(handler)

        if path == "/api/auth/password/change" and method == "POST":
            return self._handle_change_password(handler)

        if path == "/api/auth/password/forgot" and method == "POST":
            if not self.ctx.get("enable_password_reset_routes", True):
                return error_response(
                    "Password reset requires email provider configuration. "
                    "Set ARAGORA_SMTP_HOST or enable an email integration to use this endpoint.",
                    501,
                )
            return self._handle_forgot_password(handler)

        if path == "/api/auth/password/reset" and method == "POST":
            if not self.ctx.get("enable_password_reset_routes", True):
                return error_response(
                    "Password reset requires email provider configuration. "
                    "Set ARAGORA_SMTP_HOST or enable an email integration to use this endpoint.",
                    501,
                )
            return self._handle_reset_password(handler)

        if path == "/api/auth/forgot-password" and method == "POST":
            if not self.ctx.get("enable_password_reset_routes", True):
                return error_response(
                    "Password reset requires email provider configuration. "
                    "Set ARAGORA_SMTP_HOST or enable an email integration to use this endpoint.",
                    501,
                )
            return self._handle_forgot_password(handler)

        if path == "/api/auth/reset-password" and method == "POST":
            if not self.ctx.get("enable_password_reset_routes", True):
                return error_response(
                    "Password reset requires email provider configuration. "
                    "Set ARAGORA_SMTP_HOST or enable an email integration to use this endpoint.",
                    501,
                )
            return self._handle_reset_password(handler)

        if path == "/api/auth/revoke" and method == "POST":
            return self._handle_revoke_token(handler)

        if path == "/api/auth/api-key":
            if method == "POST":
                return self._handle_generate_api_key(handler)
            elif method == "DELETE":
                return self._handle_revoke_api_key(handler)

        if path == "/api/auth/api-keys":
            if method == "GET":
                return self._handle_list_api_keys(handler)
            if method == "POST":
                return self._handle_generate_api_key(handler)
            if method == "DELETE":
                return self._handle_revoke_api_key(handler)

        if path.startswith("/api/auth/api-keys/") and method == "DELETE":
            prefix = path.split("/")[-1]
            return self._handle_revoke_api_key_prefix(handler, prefix)

        # MFA endpoints
        if path == "/api/auth/mfa/setup" and method == "POST":
            return self._handle_mfa_setup(handler)

        if path == "/api/auth/mfa/enable" and method == "POST":
            return self._handle_mfa_enable(handler)

        if path == "/api/auth/mfa/disable" and method == "POST":
            return self._handle_mfa_disable(handler)

        if path == "/api/auth/mfa" and method == "DELETE":
            return self._handle_mfa_disable(handler)

        if path == "/api/auth/mfa/verify" and method == "POST":
            return self._handle_mfa_verify(handler)

        if path == "/api/auth/mfa/backup-codes" and method == "POST":
            return self._handle_mfa_backup_codes(handler)

        # Session management endpoints
        if path == "/api/auth/sessions" and method == "GET":
            return self._handle_list_sessions(handler)

        if path.startswith("/api/auth/sessions/") and method == "DELETE":
            session_id = path.split("/")[-1]
            return self._handle_revoke_session(handler, session_id)

        if path == "/api/auth/verify-email" and method == "POST":
            data = self.read_json_body(handler) or {}
            return await handle_verify_email(data)

        if (
            path in ("/api/auth/verify-email/resend", "/api/auth/resend-verification")
            and method == "POST"
        ):
            data = self.read_json_body(handler) or {}
            return await handle_resend_verification(data)

        if path == "/api/auth/setup-organization" and method == "POST":
            data = self.read_json_body(handler) or {}
            user_id, err = self._require_user_id(handler)
            if err:
                return err
            return await handle_setup_organization(data, user_id=user_id)

        if path == "/api/auth/invite" and method == "POST":
            data = self.read_json_body(handler) or {}
            user_id, err = self._require_user_id(handler)
            if err:
                return err
            return await handle_invite(data, user_id=user_id)

        if path == "/api/auth/check-invite" and method == "GET":
            return await handle_check_invite(query_params or {})

        if path == "/api/auth/accept-invite" and method == "POST":
            data = self.read_json_body(handler) or {}
            user_id, err = self._require_user_id(handler)
            if err:
                return err
            return await handle_accept_invite(data, user_id=user_id)

        if path == "/api/auth/health" and method == "GET":
            return self._handle_health(handler)

        return error_response("Method not allowed", 405)

    def _require_user_id(self, handler: Any) -> tuple[str | None, HandlerResult | None]:
        """Return authenticated user_id or error response."""
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated or not auth_ctx.user_id:
            return None, error_response("Authentication required", 401)
        return auth_ctx.user_id, None

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        return self.ctx.get("user_store")

    def _get_lockout_tracker(self) -> Any:
        """Get lockout tracker for login throttling."""
        from aragora.server.handlers.auth import login as login_module

        login_tracker = getattr(login_module, "get_lockout_tracker", None)
        handler_tracker = get_lockout_tracker

        if handler_tracker is not _ORIGINAL_LOCKOUT_TRACKER and callable(handler_tracker):
            return handler_tracker()
        if login_tracker is not _ORIGINAL_LOCKOUT_TRACKER and callable(login_tracker):
            return login_tracker()
        if callable(handler_tracker):
            return handler_tracker()
        if callable(login_tracker):
            return login_tracker()
        return _ORIGINAL_LOCKOUT_TRACKER()

    def _check_permission(
        self, handler: Any, permission_key: str, resource_id: str | None = None
    ) -> HandlerResult | None:
        """Check RBAC permission. Returns error response if denied, None if allowed.

        This builds an AuthorizationContext from the JWT token and checks
        the specified permission using the RBAC system.
        """
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        # Not authenticated - return 401
        if not auth_ctx.is_authenticated or not auth_ctx.user_id:
            return error_response("Authentication required", 401)

        # Build RBAC authorization context
        roles = {auth_ctx.role} if auth_ctx.role else {"member"}
        permissions: set[str] = set()
        for role in roles:
            permissions |= get_role_permissions(role, include_inherited=True)

        rbac_context = AuthorizationContext(
            user_id=auth_ctx.user_id,
            org_id=auth_ctx.org_id,
            roles=roles,
            permissions=permissions,
            ip_address=auth_ctx.client_ip,
        )

        # Check permission
        decision = check_permission(rbac_context, permission_key, resource_id)
        if not decision.allowed:
            logger.warning(
                f"Permission denied: user={auth_ctx.user_id} permission={permission_key} reason={decision.reason}"
            )
            return error_response(f"Permission denied: {decision.reason}", 403)

        return None  # Allowed

    # =========================================================================
    # Login/Register - Delegated to login.py
    # =========================================================================

    def _handle_register(self, handler: Any) -> HandlerResult:
        """Handle user registration."""
        return handle_register(self, handler)

    def _handle_login(self, handler: Any) -> HandlerResult:
        """Handle user login."""
        return handle_login(self, handler)

    # =========================================================================
    # Token Management - Kept in handler.py
    # =========================================================================

    @rate_limit(requests_per_minute=20, limiter_name="auth_refresh")
    @handle_errors("token refresh")
    def _handle_refresh(self, handler: Any) -> HandlerResult:
        """Handle token refresh."""
        from aragora.billing.jwt_auth import (
            create_token_pair,
            get_token_blacklist,
        )

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        refresh_token = body.get("refresh_token", "")
        if not refresh_token:
            return error_response("Refresh token required", 400)

        # Validate refresh token
        payload = validate_refresh_token(refresh_token)
        if not payload:
            return error_response("Invalid or expired refresh token", 401)

        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Get user to ensure they still exist and are active (use async if available)
        get_user_by_id = getattr(user_store, "get_user_by_id", None)
        if callable(get_user_by_id):
            user = get_user_by_id(payload.user_id)
        elif hasattr(user_store, "get_user_by_id_async"):
            user = _run_maybe_async(user_store.get_user_by_id_async(payload.user_id))
        else:
            user = None
        if not user:
            return error_response("User not found", 401)

        # 401: account disabled means the user cannot authenticate
        if not user.is_active:
            return error_response("Account is disabled", 401)

        # Revoke the old refresh token to prevent reuse
        # IMPORTANT: Persist first, then in-memory. This ensures atomic revocation:
        # If persistent fails, in-memory stays valid (fail-safe)
        # If persistent succeeds but in-memory fails, persistent check catches it
        from aragora.billing.jwt_auth import revoke_token_persistent

        try:
            revoke_token_persistent(refresh_token)
        except Exception as e:
            logger.error(f"Failed to persist token revocation: {e}")
            return error_response("Token revocation failed, please try again", 500)

        # Now update in-memory blacklist (fast local checks)
        blacklist = get_token_blacklist()
        blacklist.revoke_token(refresh_token)

        # Create new token pair
        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )

        return json_response({"tokens": tokens.to_dict()})

    @rate_limit(requests_per_minute=10, limiter_name="auth_logout")
    @handle_errors("logout")
    def _handle_logout(self, handler: Any) -> HandlerResult:
        """Handle user logout (token invalidation)."""
        # RBAC check: authentication.revoke permission required
        if error := self._check_permission(handler, "authentication.revoke"):
            return error

        from aragora.billing.jwt_auth import (
            get_token_blacklist,
            revoke_token_persistent,
        )
        from aragora.server.middleware.auth import extract_token

        # Get current user (already verified by _check_permission)
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        # Revoke the current token using both persistent and in-memory blacklists
        # IMPORTANT: Persist first, then in-memory for atomic revocation
        token = extract_token(handler)
        if token:
            # Persistent first for multi-instance consistency
            persistent_ok = revoke_token_persistent(token)

            # In-memory second for fast local checks
            blacklist = get_token_blacklist()
            in_memory_ok = blacklist.revoke_token(token)

            if persistent_ok and in_memory_ok:
                logger.info(
                    f"User logged out and token revoked (persistent + in-memory): {auth_ctx.user_id}"
                )
            elif persistent_ok:
                logger.warning(
                    f"User logged out, persistent revoked but in-memory failed: {auth_ctx.user_id}"
                )
            else:
                logger.warning(
                    f"User logged out but persistent revocation failed: {auth_ctx.user_id}"
                )
        else:
            logger.info(f"User logged out (no token to revoke): {auth_ctx.user_id}")

        # Audit log: logout
        if AUDIT_AVAILABLE and audit_logout:
            audit_logout(auth_ctx.user_id)

        return json_response({"message": "Logged out successfully"})

    @rate_limit(requests_per_minute=3, limiter_name="auth_logout_all")
    @handle_errors("logout all devices")
    @log_request("logout all devices")
    def _handle_logout_all(self, handler: Any) -> HandlerResult:
        """
        Handle logout from all devices.

        Increments the user's token_version, immediately invalidating all
        existing JWT tokens for this user across all devices.
        Also revokes the current token for immediate effect.
        """
        # RBAC check: authentication.revoke permission required
        if error := self._check_permission(handler, "authentication.revoke"):
            return error

        from aragora.billing.jwt_auth import (
            get_token_blacklist,
            revoke_token_persistent,
        )
        from aragora.server.middleware.auth import extract_token

        # Get current user (already verified by _check_permission)
        user_store = self._get_user_store()
        if not user_store:
            return error_response("User service unavailable", 503)

        auth_ctx = extract_user_from_request(handler, user_store)

        # Increment token version to invalidate all existing tokens
        new_version = user_store.increment_token_version(auth_ctx.user_id)
        if new_version == 0:
            return error_response("User not found", 404)

        # Also revoke current token for immediate effect (before version check)
        token = extract_token(handler)
        if token:
            blacklist = get_token_blacklist()
            blacklist.revoke_token(token)
            revoke_token_persistent(token)

        logger.info(f"logout_all user_id={auth_ctx.user_id} new_token_version={new_version}")

        # Audit log: logout all sessions
        if AUDIT_AVAILABLE and audit_security:
            audit_security(
                event_type="anomaly",
                actor_id=auth_ctx.user_id,
                reason="all_sessions_terminated",
            )

        return json_response(
            {
                "message": "All sessions terminated",
                "sessions_invalidated": True,
                "token_version": new_version,
            }
        )

    # =========================================================================
    # Profile Methods - Kept in handler.py
    # =========================================================================

    # Cache-control headers to prevent CDN caching of auth responses
    AUTH_NO_CACHE_HEADERS = {
        "Cache-Control": "no-store, no-cache, must-revalidate, private",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    def _handle_health(self, handler: Any) -> HandlerResult:
        """Lightweight diagnostic endpoint - no auth or DB required."""
        import time

        info: dict[str, Any] = {"status": "ok", "timestamp": time.time()}

        # Pool status
        try:
            from aragora.storage.pool_manager import (
                get_shared_pool,
                is_pool_initialized,
                get_pool_event_loop,
            )

            pool = get_shared_pool() if is_pool_initialized() else None
            main_loop = get_pool_event_loop()
            info["pool"] = {
                "initialized": is_pool_initialized(),
                "size": getattr(pool, "get_size", lambda: None)() if pool else None,
                "free": getattr(pool, "get_idle_size", lambda: None)() if pool else None,
                "main_loop_running": main_loop.is_running() if main_loop else None,
                "main_loop_id": id(main_loop) if main_loop else None,
            }
        except Exception as e:
            info["pool"] = {"error": str(e)}

        # JWT decode check (no DB needed)
        try:
            from aragora.server.middleware.auth import extract_token
            from aragora.billing.jwt_auth import decode_jwt

            token = extract_token(handler)
            if token:
                payload = decode_jwt(token)
                info["jwt"] = {
                    "valid": payload is not None,
                    "user_id": getattr(payload, "user_id", None) if payload else None,
                }
            else:
                info["jwt"] = {"provided": False}
        except Exception as e:
            info["jwt"] = {"error": str(e)}

        return json_response(info, headers=self.AUTH_NO_CACHE_HEADERS)

    @rate_limit(requests_per_minute=30, limiter_name="auth_get_me")
    @handle_errors("get user info")
    async def _handle_get_me(self, handler: Any) -> HandlerResult:
        """Get current user information."""
        logger.info("[/me] Step 1: Checking authentication.read permission")
        # RBAC check: authentication.read permission required
        if error := self._check_permission(handler, "authentication.read"):
            logger.warning("[/me] Permission check failed")
            return error

        # Get current user (already verified by _check_permission)
        logger.info("[/me] Step 2: Permission OK, extracting auth context")
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        logger.info(
            "[/me] Step 3: Auth context: authenticated=%s, user_id=%s",
            auth_ctx.is_authenticated,
            auth_ctx.user_id,
        )

        # Get user store
        if not user_store:
            logger.error("[/me] No user_store in context")
            return error_response(
                "Authentication service unavailable", 503, headers=self.AUTH_NO_CACHE_HEADERS
            )

        # Get full user data using async methods to avoid run_async() on a
        # running event loop (this coroutine is awaited from async handle()).
        logger.info("[/me] Step 4: Looking up user by ID")
        get_by_id_async = getattr(user_store, "get_user_by_id_async", None)
        if get_by_id_async and asyncio.iscoroutinefunction(get_by_id_async):
            user = await get_by_id_async(auth_ctx.user_id)
        else:
            get_user_by_id = getattr(user_store, "get_user_by_id", None)
            user = get_user_by_id(auth_ctx.user_id) if callable(get_user_by_id) else None
        logger.info("[/me] Step 5: User lookup result: %s", "found" if user else "not found")
        if not user:
            return error_response("User not found", 404, headers=self.AUTH_NO_CACHE_HEADERS)

        # Get organization if user belongs to one
        org_data = None
        if user.org_id:
            get_org_async = getattr(user_store, "get_organization_by_id_async", None)
            if get_org_async and asyncio.iscoroutinefunction(get_org_async):
                org = await get_org_async(user.org_id)
            else:
                get_org_by_id = getattr(user_store, "get_organization_by_id", None)
                org = get_org_by_id(user.org_id) if callable(get_org_by_id) else None
            if org:
                org_data = org.to_dict()

        return json_response(
            {
                "user": user.to_dict(),
                "organization": org_data,
            },
            headers=self.AUTH_NO_CACHE_HEADERS,
        )

    @rate_limit(requests_per_minute=5, limiter_name="auth_update_me")
    @handle_errors("update user info")
    def _handle_update_me(self, handler: Any) -> HandlerResult:
        """Update current user information."""
        # RBAC check: authentication.read permission required (user updating own info)
        if error := self._check_permission(handler, "authentication.read"):
            return error

        # Get current user (already verified by _check_permission)
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Get user store
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Get user (use async method if available)
        get_user_by_id = getattr(user_store, "get_user_by_id", None)
        if callable(get_user_by_id):
            user = get_user_by_id(auth_ctx.user_id)
        elif hasattr(user_store, "get_user_by_id_async"):
            user = _run_maybe_async(user_store.get_user_by_id_async(auth_ctx.user_id))
        else:
            user = None
        if not user:
            return error_response("User not found", 404)

        # Update allowed fields
        updates = {}
        if "name" in body:
            updates["name"] = str(body["name"]).strip()[:100]

        # Save updates
        if updates:
            update_user = getattr(user_store, "update_user", None)
            if callable(update_user):
                update_user(user.id, **updates)
                user = update_user and user_store.get_user_by_id(user.id)
            elif hasattr(user_store, "update_user_async"):
                _run_maybe_async(user_store.update_user_async(user.id, **updates))
                user = _run_maybe_async(user_store.get_user_by_id_async(user.id))

        return json_response({"user": user.to_dict()})

    # =========================================================================
    # Token Revocation
    # =========================================================================

    @rate_limit(requests_per_minute=10, limiter_name="auth_revoke_token")
    @handle_errors("revoke token")
    def _handle_revoke_token(self, handler: Any) -> HandlerResult:
        """Explicitly revoke a specific token."""
        # RBAC check: session.revoke permission required
        if error := self._check_permission(handler, "session.revoke"):
            return error

        from aragora.billing.jwt_auth import (
            extract_user_from_request,
            get_token_blacklist,
            revoke_token_persistent,
        )
        from aragora.server.middleware.auth import extract_token

        # Get current user (already verified by _check_permission)
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)

        # Parse request body
        body = self.read_json_body(handler)

        # Get token to revoke from body, or use current token
        token_to_revoke = None
        if body and "token" in body:
            token_to_revoke = body["token"]
        else:
            token_to_revoke = extract_token(handler)

        if not token_to_revoke:
            return error_response("No token provided to revoke", 400)

        # Revoke using both in-memory (fast) and persistent (multi-instance) blacklists
        blacklist = get_token_blacklist()
        in_memory_ok = blacklist.revoke_token(token_to_revoke)
        persistent_ok = revoke_token_persistent(token_to_revoke)

        if in_memory_ok:
            if persistent_ok:
                logger.info(f"Token revoked (in-memory + persistent) by user: {auth_ctx.user_id}")
            else:
                logger.warning(
                    f"Token revoked in-memory but persistent failed for user: {auth_ctx.user_id}"
                )
            return json_response(
                {
                    "message": "Token revoked successfully",
                    "blacklist_size": blacklist.size(),
                    "persistent": persistent_ok,
                }
            )
        else:
            return error_response("Invalid token - could not revoke", 400)

    # =========================================================================
    # Password Management - Delegated to password.py
    # =========================================================================

    def _handle_change_password(self, handler: Any) -> HandlerResult:
        """Change user password."""
        return handle_change_password(self, handler)

    def _handle_forgot_password(self, handler: Any) -> HandlerResult:
        """Handle forgot password request."""
        return handle_forgot_password(self, handler)

    def _handle_reset_password(self, handler: Any) -> HandlerResult:
        """Handle password reset with token."""
        return handle_reset_password(self, handler)

    def _send_password_reset_email(self, user: Any, reset_link: str) -> None:
        """Send password reset email to user (fire-and-forget)."""
        send_password_reset_email(user, reset_link)

    # =========================================================================
    # API Key Management - Delegated to api_keys.py
    # =========================================================================

    def _handle_generate_api_key(self, handler: Any) -> HandlerResult:
        """Generate a new API key for the user."""
        return handle_generate_api_key(self, handler)

    def _handle_revoke_api_key(self, handler: Any) -> HandlerResult:
        """Revoke the user's API key."""
        return handle_revoke_api_key(self, handler)

    def _handle_list_api_keys(self, handler: Any) -> HandlerResult:
        """List API keys for the current user."""
        return handle_list_api_keys(self, handler)

    def _handle_revoke_api_key_prefix(self, handler: Any, prefix: str) -> HandlerResult:
        """Revoke the user's API key by prefix."""
        return handle_revoke_api_key_prefix(self, handler, prefix)

    # =========================================================================
    # MFA/2FA Methods - Delegated to mfa.py
    # =========================================================================

    def _handle_mfa_setup(self, handler: Any) -> HandlerResult:
        """Generate MFA secret and provisioning URI for setup."""
        return handle_mfa_setup(self, handler)

    def _handle_mfa_enable(self, handler: Any) -> HandlerResult:
        """Enable MFA after verifying setup code."""
        return handle_mfa_enable(self, handler)

    def _handle_mfa_disable(self, handler: Any) -> HandlerResult:
        """Disable MFA for the user."""
        return handle_mfa_disable(self, handler)

    def _handle_mfa_verify(self, handler: Any) -> HandlerResult:
        """Verify MFA code during login."""
        return handle_mfa_verify(self, handler)

    def _handle_mfa_backup_codes(self, handler: Any) -> HandlerResult:
        """Regenerate MFA backup codes."""
        return handle_mfa_backup_codes(self, handler)

    # =========================================================================
    # Session Management - Delegated to sessions.py
    # =========================================================================

    def _handle_list_sessions(self, handler: Any) -> HandlerResult:
        """List all active sessions for the current user."""
        return handle_list_sessions(self, handler)

    def _handle_revoke_session(self, handler: Any, session_id: str) -> HandlerResult:
        """Revoke a specific session."""
        return handle_revoke_session(self, handler, session_id)


__all__ = ["AuthHandler"]
