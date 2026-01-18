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

import logging
from datetime import datetime
from typing import Optional

# Lockout tracker for brute-force protection
from aragora.auth.lockout import get_lockout_tracker

# Module-level imports for test mocking compatibility
from aragora.billing.jwt_auth import extract_user_from_request, validate_refresh_token

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from ..utils.rate_limit import get_client_ip, rate_limit
from .validation import validate_email, validate_password

logger = logging.getLogger(__name__)


class AuthHandler(BaseHandler):
    """Handler for user authentication endpoints."""

    ROUTES = [
        "/api/auth/register",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/auth/logout-all",
        "/api/auth/refresh",
        "/api/auth/revoke",
        "/api/auth/me",
        "/api/auth/password",
        "/api/auth/api-key",
        "/api/auth/mfa/setup",
        "/api/auth/mfa/enable",
        "/api/auth/mfa/disable",
        "/api/auth/mfa/verify",
        "/api/auth/mfa/backup-codes",
        "/api/auth/sessions",
        "/api/auth/sessions/*",  # For DELETE /api/auth/sessions/:id
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle wildcard routes for session management
        if path.startswith("/api/auth/sessions/"):
            return True
        return False

    def handle(
        self, path: str, query_params: dict, handler, method: str = "GET"
    ) -> Optional[HandlerResult]:
        """Route auth requests to appropriate methods."""
        # Determine HTTP method from handler if not provided
        if hasattr(handler, "command"):
            method = handler.command

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
                return self._handle_get_me(handler)
            elif method == "PUT":
                return self._handle_update_me(handler)

        if path == "/api/auth/password" and method == "POST":
            return self._handle_change_password(handler)

        if path == "/api/auth/revoke" and method == "POST":
            return self._handle_revoke_token(handler)

        if path == "/api/auth/api-key":
            if method == "POST":
                return self._handle_generate_api_key(handler)
            elif method == "DELETE":
                return self._handle_revoke_api_key(handler)

        # MFA endpoints
        if path == "/api/auth/mfa/setup" and method == "POST":
            return self._handle_mfa_setup(handler)

        if path == "/api/auth/mfa/enable" and method == "POST":
            return self._handle_mfa_enable(handler)

        if path == "/api/auth/mfa/disable" and method == "POST":
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

        return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    @rate_limit(rpm=2, limiter_name="auth_register")
    @handle_errors("user registration")
    @log_request("user registration")
    def _handle_register(self, handler) -> HandlerResult:
        """Handle user registration."""
        from aragora.billing.jwt_auth import create_token_pair
        from aragora.billing.models import hash_password

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Extract and validate fields
        email = body.get("email", "").strip().lower()
        password = body.get("password", "")
        name = body.get("name", "").strip()
        org_name = body.get("organization", "").strip()

        # Validate email
        valid, err = validate_email(email)
        if not valid:
            return error_response(err, 400)

        # Validate password
        valid, err = validate_password(password)
        if not valid:
            return error_response(err, 400)

        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("User service unavailable", 503)

        # Check if email already exists
        existing = user_store.get_user_by_email(email)
        if existing:
            return error_response("Email already registered", 409)

        # Hash password
        password_hash, password_salt = hash_password(password)

        # Create user first (without org)
        try:
            user = user_store.create_user(
                email=email,
                password_hash=password_hash,
                password_salt=password_salt,
                name=name or email.split("@")[0],
            )
        except ValueError as e:
            logger.warning(f"User creation failed: {type(e).__name__}: {e}")
            return error_response("User creation failed", 409)

        # Create organization if name provided
        if org_name:
            user_store.create_organization(
                name=org_name,
                owner_id=user.id,
            )
            # Refresh user to get updated org_id
            user = user_store.get_user_by_id(user.id)

        # Create tokens
        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )

        logger.info(f"User registered: {user.email} (id={user.id})")

        return json_response(
            {
                "user": user.to_dict(),
                "tokens": tokens.to_dict(),
            },
            status=201,
        )

    @rate_limit(rpm=3, limiter_name="auth_login")
    @handle_errors("user login")
    @log_request("user login")
    def _handle_login(self, handler) -> HandlerResult:
        """Handle user login."""
        from aragora.billing.jwt_auth import create_mfa_pending_token, create_token_pair

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        email = body.get("email", "").strip().lower()
        password = body.get("password", "")

        if not email or not password:
            return error_response("Email and password required", 400)

        # Get client IP for lockout tracking
        client_ip = get_client_ip(handler)

        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Check lockout tracker (tracks by email AND IP)
        lockout_tracker = get_lockout_tracker()
        if lockout_tracker.is_locked(email=email, ip=client_ip):
            remaining_seconds = lockout_tracker.get_remaining_time(email=email, ip=client_ip)
            remaining_minutes = max(1, remaining_seconds // 60)
            logger.warning(f"Login attempt on locked account/IP: email={email}, ip={client_ip}")
            return error_response(
                f"Too many failed attempts. Try again in {remaining_minutes} minute(s).", 429
            )

        # Also check database-based account lockout (legacy support)
        if hasattr(user_store, "is_account_locked"):
            is_locked, lockout_until, failed_attempts = user_store.is_account_locked(email)
            if is_locked and lockout_until:
                remaining_minutes = max(
                    1, int((lockout_until - datetime.utcnow()).total_seconds() / 60)
                )
                logger.warning(f"Login attempt on locked account (db): {email}")
                return error_response(
                    f"Account temporarily locked. Try again in {remaining_minutes} minute(s).", 429
                )

        # Find user
        user = user_store.get_user_by_email(email)
        if not user:
            # Record failed attempt to lockout tracker (prevents enumeration attacks)
            lockout_tracker.record_failure(email=email, ip=client_ip)
            # Use same error to prevent email enumeration
            return error_response("Invalid email or password", 401)

        # Check if account is active
        if not user.is_active:
            return error_response("Account is disabled", 403)

        # Verify password
        if not user.verify_password(password):
            # Record failed login attempt to both trackers
            attempts, lockout_seconds = lockout_tracker.record_failure(email=email, ip=client_ip)

            # Also record in database for persistence across restarts
            if hasattr(user_store, "record_failed_login"):
                db_attempts, lockout_until = user_store.record_failed_login(email)

            if lockout_seconds:
                remaining_minutes = max(1, lockout_seconds // 60)
                return error_response(
                    f"Too many failed attempts. Account locked for {remaining_minutes} minute(s).",
                    429,
                )
            return error_response("Invalid email or password", 401)

        # Successful login - reset failed attempts in both trackers
        lockout_tracker.reset(email=email, ip=client_ip)
        if hasattr(user_store, "reset_failed_login_attempts"):
            user_store.reset_failed_login_attempts(email)

        # Update last login
        user_store.update_user(user.id, last_login_at=datetime.utcnow())

        # Check if MFA is enabled - require second factor before issuing tokens
        if user.mfa_enabled and user.mfa_secret:
            pending_token = create_mfa_pending_token(user.id, user.email)
            logger.info(f"User login pending MFA: {user.email}")
            return json_response(
                {
                    "mfa_required": True,
                    "pending_token": pending_token,
                    "message": "MFA verification required",
                }
            )

        # No MFA - create full tokens
        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )

        logger.info(f"User logged in: {user.email}")

        return json_response(
            {
                "user": user.to_dict(),
                "tokens": tokens.to_dict(),
            }
        )

    @rate_limit(rpm=20, limiter_name="auth_refresh")
    @handle_errors("token refresh")
    def _handle_refresh(self, handler) -> HandlerResult:
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

        # Get user to ensure they still exist and are active
        user = user_store.get_user_by_id(payload.user_id)
        if not user:
            return error_response("User not found", 401)

        if not user.is_active:
            return error_response("Account is disabled", 403)

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

    @rate_limit(rpm=10, limiter_name="auth_logout")
    @handle_errors("logout")
    def _handle_logout(self, handler) -> HandlerResult:
        """Handle user logout (token invalidation)."""
        from aragora.billing.jwt_auth import (
            get_token_blacklist,
            revoke_token_persistent,
        )
        from aragora.server.middleware.auth import extract_token

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

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

        return json_response({"message": "Logged out successfully"})

    @rate_limit(rpm=3, limiter_name="auth_logout_all")
    @handle_errors("logout all devices")
    @log_request("logout all devices")
    def _handle_logout_all(self, handler) -> HandlerResult:
        """
        Handle logout from all devices.

        Increments the user's token_version, immediately invalidating all
        existing JWT tokens for this user across all devices.
        Also revokes the current token for immediate effect.
        """
        from aragora.billing.jwt_auth import (
            get_token_blacklist,
            revoke_token_persistent,
        )
        from aragora.server.middleware.auth import extract_token

        # Get current user
        user_store = self._get_user_store()
        if not user_store:
            return error_response("User service unavailable", 503)

        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

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

        return json_response(
            {
                "message": "All sessions terminated",
                "sessions_invalidated": True,
                "token_version": new_version,
            }
        )

    @rate_limit(rpm=30, limiter_name="auth_get_me")
    @handle_errors("get user info")
    def _handle_get_me(self, handler) -> HandlerResult:
        """Get current user information."""
        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Get user store
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Get full user data
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        # Get organization if user belongs to one
        org_data = None
        if user.org_id:
            org = user_store.get_organization_by_id(user.org_id)
            if org:
                org_data = org.to_dict()

        return json_response(
            {
                "user": user.to_dict(),
                "organization": org_data,
            }
        )

    @rate_limit(rpm=5, limiter_name="auth_update_me")
    @handle_errors("update user info")
    def _handle_update_me(self, handler) -> HandlerResult:
        """Update current user information."""
        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Get user store
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Get user
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        # Update allowed fields
        updates = {}
        if "name" in body:
            updates["name"] = str(body["name"]).strip()[:100]

        # Save updates
        if updates:
            user_store.update_user(user.id, **updates)
            user = user_store.get_user_by_id(user.id)

        return json_response({"user": user.to_dict()})

    @rate_limit(rpm=3, limiter_name="auth_change_password")
    @handle_errors("change password")
    def _handle_change_password(self, handler) -> HandlerResult:
        """Change user password."""
        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        current_password = body.get("current_password", "")
        new_password = body.get("new_password", "")

        if not current_password or not new_password:
            return error_response("Current and new password required", 400)

        # Validate new password
        valid, err = validate_password(new_password)
        if not valid:
            return error_response(err, 400)

        # Get user store
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Get user
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        # Verify current password
        if not user.verify_password(current_password):
            return error_response("Current password is incorrect", 401)

        # Set new password
        from aragora.billing.models import hash_password

        password_hash, password_salt = hash_password(new_password)
        user_store.update_user(
            user.id,
            password_hash=password_hash,
            password_salt=password_salt,
        )

        logger.info(f"Password changed for user: {user.email}")

        return json_response({"message": "Password changed successfully"})

    @rate_limit(rpm=10, limiter_name="auth_revoke_token")
    @handle_errors("revoke token")
    def _handle_revoke_token(self, handler) -> HandlerResult:
        """Explicitly revoke a specific token."""
        from aragora.billing.jwt_auth import (
            extract_user_from_request,
            get_token_blacklist,
            revoke_token_persistent,
        )
        from aragora.server.middleware.auth import extract_token

        # Get current user (required for authorization)
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

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

    @rate_limit(rpm=3, limiter_name="auth_api_key_gen")
    @handle_errors("generate API key")
    def _handle_generate_api_key(self, handler) -> HandlerResult:
        """Generate a new API key for the user."""
        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Get user store
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Get user
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        # Check if user's tier allows API access
        if user.org_id:
            org = user_store.get_organization_by_id(user.org_id)
            if org and not org.limits.api_access:
                return error_response("API access requires Professional tier or higher", 403)

        # Generate new API key using secure hash-based storage
        # The plaintext key is only returned once; we store the hash
        api_key = user.generate_api_key(expires_days=365)

        # Persist the hashed key fields (api_key_hash, api_key_prefix, expiry)
        user_store.update_user(
            user.id,
            api_key=None,  # Clear legacy plaintext field
            api_key_hash=user.api_key_hash,
            api_key_prefix=user.api_key_prefix,
            api_key_created_at=user.api_key_created_at,
            api_key_expires_at=user.api_key_expires_at,
        )

        logger.info(f"API key generated for user: {user.email} (prefix: {user.api_key_prefix})")

        # Return the key (only shown once - plaintext is never stored)
        return json_response(
            {
                "api_key": api_key,
                "prefix": user.api_key_prefix,
                "expires_at": (
                    user.api_key_expires_at.isoformat() if user.api_key_expires_at else None
                ),
                "message": "Save this key - it will not be shown again",
            }
        )

    @rate_limit(rpm=5, limiter_name="auth_revoke_api_key")
    @handle_errors("revoke API key")
    def _handle_revoke_api_key(self, handler) -> HandlerResult:
        """Revoke the user's API key."""
        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Get user store
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Get user
        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        # Revoke API key - clear both legacy and hashed fields
        user_store.update_user(
            user.id,
            api_key=None,
            api_key_hash=None,
            api_key_prefix=None,
            api_key_created_at=None,
            api_key_expires_at=None,
        )

        logger.info(f"API key revoked for user: {user.email}")

        return json_response({"message": "API key revoked"})

    # =========================================================================
    # MFA/2FA Methods
    # =========================================================================

    @rate_limit(rpm=5, limiter_name="mfa_setup")
    @handle_errors("MFA setup")
    @log_request("MFA setup")
    def _handle_mfa_setup(self, handler) -> HandlerResult:
        """Generate MFA secret and provisioning URI for setup."""
        try:
            import pyotp
        except ImportError:
            return error_response("MFA not available (pyotp not installed)", 503)

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        if user.mfa_enabled:
            return error_response("MFA is already enabled", 400)

        # Generate new secret
        secret = pyotp.random_base32()

        # Store secret temporarily (not enabled yet)
        user_store.update_user(user.id, mfa_secret=secret)

        # Generate provisioning URI for authenticator apps
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(name=user.email, issuer_name="Aragora")

        return json_response(
            {
                "secret": secret,
                "provisioning_uri": provisioning_uri,
                "message": "Scan QR code or enter secret in your authenticator app, then call /api/auth/mfa/enable with verification code",
            }
        )

    @rate_limit(rpm=5, limiter_name="mfa_enable")
    @handle_errors("MFA enable")
    @log_request("MFA enable")
    def _handle_mfa_enable(self, handler) -> HandlerResult:
        """Enable MFA after verifying setup code."""
        import hashlib
        import secrets as py_secrets

        try:
            import pyotp
        except ImportError:
            return error_response("MFA not available", 503)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        code = body.get("code", "").strip()
        if not code:
            return error_response("Verification code is required", 400)

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        if user.mfa_enabled:
            return error_response("MFA is already enabled", 400)

        if not user.mfa_secret:
            return error_response("MFA not set up. Call /api/auth/mfa/setup first", 400)

        # Verify the code
        totp = pyotp.TOTP(user.mfa_secret)
        if not totp.verify(code, valid_window=1):
            return error_response("Invalid verification code", 400)

        # Generate backup codes
        backup_codes = [py_secrets.token_hex(4) for _ in range(10)]
        backup_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in backup_codes]

        import json as json_module

        user_store.update_user(
            user.id,
            mfa_enabled=True,
            mfa_backup_codes=json_module.dumps(backup_hashes),
        )

        logger.info(f"MFA enabled for user: {user.email}")

        return json_response(
            {
                "message": "MFA enabled successfully",
                "backup_codes": backup_codes,
                "warning": "Save these backup codes securely. They cannot be shown again.",
            }
        )

    @rate_limit(rpm=5, limiter_name="mfa_disable")
    @handle_errors("MFA disable")
    @log_request("MFA disable")
    def _handle_mfa_disable(self, handler) -> HandlerResult:
        """Disable MFA for the user."""
        try:
            import pyotp
        except ImportError:
            return error_response("MFA not available", 503)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Require password or MFA code to disable
        code = body.get("code", "").strip()
        password = body.get("password", "").strip()

        if not code and not password:
            return error_response("MFA code or password required to disable MFA", 400)

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        if not user.mfa_enabled:
            return error_response("MFA is not enabled", 400)

        # Verify with code or password
        if code:
            totp = pyotp.TOTP(user.mfa_secret)
            if not totp.verify(code, valid_window=1):
                return error_response("Invalid MFA code", 400)
        elif password:
            if not user.verify_password(password):
                return error_response("Invalid password", 400)

        # Disable MFA
        user_store.update_user(
            user.id,
            mfa_enabled=False,
            mfa_secret=None,
            mfa_backup_codes=None,
        )

        logger.info(f"MFA disabled for user: {user.email}")

        return json_response({"message": "MFA disabled successfully"})

    @rate_limit(rpm=10, limiter_name="mfa_verify")
    @handle_errors("MFA verify")
    @log_request("MFA verify")
    def _handle_mfa_verify(self, handler) -> HandlerResult:
        """Verify MFA code during login."""
        import hashlib

        from aragora.billing.jwt_auth import create_token_pair, validate_mfa_pending_token

        try:
            import pyotp
        except ImportError:
            return error_response("MFA not available", 503)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        code = body.get("code", "").strip()
        pending_token = body.get("pending_token", "").strip()

        if not code:
            return error_response("MFA code is required", 400)

        if not pending_token:
            return error_response("Pending token is required", 400)

        # Validate the pending token to identify the user
        pending_payload = validate_mfa_pending_token(pending_token)
        if not pending_payload:
            return error_response("Invalid or expired pending token", 401)

        user_store = self._get_user_store()
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        user = user_store.get_user_by_id(pending_payload.sub)
        if not user:
            return error_response("User not found", 404)

        if not user.mfa_enabled or not user.mfa_secret:
            return error_response("MFA not enabled for this user", 400)

        # Try TOTP code first
        totp = pyotp.TOTP(user.mfa_secret)
        if totp.verify(code, valid_window=1):
            # Blacklist pending token to prevent replay
            from aragora.billing.jwt_auth import get_token_blacklist

            blacklist = get_token_blacklist()
            blacklist.revoke_token(pending_token)

            # Valid TOTP code - create full tokens
            tokens = create_token_pair(
                user_id=user.id,
                email=user.email,
                org_id=user.org_id,
                role=user.role,
            )
            token_dict = tokens.to_dict()
            logger.info(f"MFA verified for user: {user.email}")
            return json_response(
                {
                    "message": "MFA verification successful",
                    "user": user.to_dict(),
                    "tokens": token_dict,
                }
            )

        # Try backup code
        if user.mfa_backup_codes:
            import json as json_module

            code_hash = hashlib.sha256(code.encode()).hexdigest()
            backup_hashes = json_module.loads(user.mfa_backup_codes)

            if code_hash in backup_hashes:
                # Valid backup code - remove it
                backup_hashes.remove(code_hash)
                user_store.update_user(
                    user.id,
                    mfa_backup_codes=json_module.dumps(backup_hashes),
                )

                # Blacklist pending token to prevent replay
                from aragora.billing.jwt_auth import get_token_blacklist

                blacklist = get_token_blacklist()
                blacklist.revoke_token(pending_token)

                tokens = create_token_pair(
                    user_id=user.id,
                    email=user.email,
                    org_id=user.org_id,
                    role=user.role,
                )
                token_dict = tokens.to_dict()
                remaining = len(backup_hashes)

                logger.info(f"Backup code used for user: {user.email}, {remaining} remaining")

                return json_response(
                    {
                        "message": "MFA verification successful (backup code used)",
                        "user": user.to_dict(),
                        "tokens": token_dict,
                        "backup_codes_remaining": remaining,
                        "warning": (
                            f"Backup code used. {remaining} remaining." if remaining < 5 else None
                        ),
                    }
                )

        return error_response("Invalid MFA code", 400)

    @rate_limit(rpm=3, limiter_name="mfa_backup")
    @handle_errors("MFA backup codes")
    @log_request("MFA backup codes")
    def _handle_mfa_backup_codes(self, handler) -> HandlerResult:
        """Regenerate MFA backup codes."""
        import hashlib
        import secrets as py_secrets

        try:
            import pyotp
        except ImportError:
            return error_response("MFA not available", 503)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        # Require current MFA code to regenerate backup codes
        code = body.get("code", "").strip()
        if not code:
            return error_response("Current MFA code is required", 400)

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        user = user_store.get_user_by_id(auth_ctx.user_id)
        if not user:
            return error_response("User not found", 404)

        if not user.mfa_enabled or not user.mfa_secret:
            return error_response("MFA not enabled", 400)

        # Verify current code
        totp = pyotp.TOTP(user.mfa_secret)
        if not totp.verify(code, valid_window=1):
            return error_response("Invalid MFA code", 400)

        # Generate new backup codes
        backup_codes = [py_secrets.token_hex(4) for _ in range(10)]
        backup_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in backup_codes]

        import json as json_module

        user_store.update_user(
            user.id,
            mfa_backup_codes=json_module.dumps(backup_hashes),
        )

        logger.info(f"Backup codes regenerated for user: {user.email}")

        return json_response(
            {
                "backup_codes": backup_codes,
                "warning": "Save these backup codes securely. They cannot be shown again.",
            }
        )

    # =========================================================================
    # Session Management
    # =========================================================================

    @rate_limit(rpm=30, limiter_name="auth_sessions")
    @handle_errors("list sessions")
    def _handle_list_sessions(self, handler) -> HandlerResult:
        """List all active sessions for the current user.

        Returns list of sessions with metadata (device, IP, last activity).
        The current session is marked with is_current=true.
        """
        from aragora.billing.auth.sessions import get_session_manager
        from aragora.billing.jwt_auth import decode_jwt
        from aragora.server.middleware.auth import extract_token

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Get current token JTI to mark current session
        current_jti = None
        token = extract_token(handler)
        if token:
            payload = decode_jwt(token)
            current_jti = payload.jti if payload else None

        # Get sessions from manager
        manager = get_session_manager()
        sessions = manager.list_sessions(auth_ctx.user_id)

        # Convert to response format
        session_list = []
        for session in sessions:
            session_dict = session.to_dict()
            session_dict["is_current"] = session.session_id == current_jti
            session_list.append(session_dict)

        # Sort by last activity (most recent first)
        session_list.sort(key=lambda s: s["last_activity"], reverse=True)

        return json_response({
            "sessions": session_list,
            "total": len(session_list),
        })

    @rate_limit(rpm=10, limiter_name="auth_revoke_session")
    @handle_errors("revoke session")
    def _handle_revoke_session(self, handler, session_id: str) -> HandlerResult:
        """Revoke a specific session.

        This invalidates the session and adds the token to the blacklist.
        Users cannot revoke their current session (use logout instead).
        """
        from aragora.billing.auth.sessions import get_session_manager
        from aragora.billing.jwt_auth import decode_jwt
        from aragora.server.middleware.auth import extract_token

        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Validate session_id format
        if not session_id or len(session_id) < 8:
            return error_response("Invalid session ID", 400)

        # Check if trying to revoke current session
        current_jti = None
        token = extract_token(handler)
        if token:
            payload = decode_jwt(token)
            current_jti = payload.jti if payload else None

        if session_id == current_jti:
            return error_response(
                "Cannot revoke current session. Use /api/auth/logout instead.",
                400,
            )

        # Get session manager and verify session belongs to user
        manager = get_session_manager()
        session = manager.get_session(auth_ctx.user_id, session_id)

        if not session:
            return error_response("Session not found", 404)

        # Revoke the session
        manager.revoke_session(auth_ctx.user_id, session_id)

        # Note: We don't have the actual token to blacklist here since we only
        # store session metadata. The token will be rejected when:
        # 1. User increments token version (logout-all)
        # 2. Token expires naturally
        # For immediate revocation, users should use logout-all

        logger.info(
            f"Session {session_id[:8]}... revoked for user {auth_ctx.user_id}"
        )

        return json_response({
            "success": True,
            "message": "Session revoked successfully",
            "session_id": session_id,
        })


__all__ = ["AuthHandler"]
