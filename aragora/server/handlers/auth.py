"""
User Authentication Handlers.

Endpoints:
- POST /api/auth/register - Create a new user account
- POST /api/auth/login - Authenticate and get tokens
- POST /api/auth/logout - Invalidate current token (adds to blacklist)
- POST /api/auth/refresh - Refresh access token (revokes old refresh token)
- POST /api/auth/revoke - Explicitly revoke a specific token
- GET /api/auth/me - Get current user information
- PUT /api/auth/me - Update current user information
- POST /api/auth/password - Change password
- POST /api/auth/api-key - Generate API key
- DELETE /api/auth/api-key - Revoke API key
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    log_request,
)
from .utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Email validation pattern
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Password requirements
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128


def validate_email(email: str) -> tuple[bool, str]:
    """Validate email format."""
    if not email:
        return False, "Email is required"
    if len(email) > 254:
        return False, "Email too long"
    if not EMAIL_PATTERN.match(email):
        return False, "Invalid email format"
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password requirements."""
    if not password:
        return False, "Password is required"
    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    if len(password) > MAX_PASSWORD_LENGTH:
        return False, f"Password must be at most {MAX_PASSWORD_LENGTH} characters"
    return True, ""


class AuthHandler(BaseHandler):
    """Handler for user authentication endpoints."""

    ROUTES = [
        "/api/auth/register",
        "/api/auth/login",
        "/api/auth/logout",
        "/api/auth/refresh",
        "/api/auth/revoke",
        "/api/auth/me",
        "/api/auth/password",
        "/api/auth/api-key",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

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

        return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    @rate_limit(rpm=5, limiter_name="auth_register")
    @handle_errors("user registration")
    @log_request("user registration")
    def _handle_register(self, handler) -> HandlerResult:
        """Handle user registration."""
        from aragora.billing.models import hash_password
        from aragora.billing.jwt_auth import create_token_pair

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
            return error_response(str(e), 409)

        # Create organization if name provided
        if org_name:
            org = user_store.create_organization(
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

    @rate_limit(rpm=10, limiter_name="auth_login")
    @handle_errors("user login")
    @log_request("user login")
    def _handle_login(self, handler) -> HandlerResult:
        """Handle user login."""
        from aragora.billing.jwt_auth import create_token_pair

        # Parse request body
        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        email = body.get("email", "").strip().lower()
        password = body.get("password", "")

        if not email or not password:
            return error_response("Email and password required", 400)

        # Get user store
        user_store = self._get_user_store()
        if not user_store:
            return error_response("Authentication service unavailable", 503)

        # Find user
        user = user_store.get_user_by_email(email)
        if not user:
            # Use same error to prevent email enumeration
            return error_response("Invalid email or password", 401)

        # Check if account is active
        if not user.is_active:
            return error_response("Account is disabled", 403)

        # Verify password
        if not user.verify_password(password):
            return error_response("Invalid email or password", 401)

        # Update last login
        user_store.update_user(user.id, last_login_at=datetime.utcnow())

        # Create tokens
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

    @handle_errors("token refresh")
    def _handle_refresh(self, handler) -> HandlerResult:
        """Handle token refresh."""
        from aragora.billing.jwt_auth import (
            validate_refresh_token,
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

    @handle_errors("logout")
    def _handle_logout(self, handler) -> HandlerResult:
        """Handle user logout (token invalidation)."""
        from aragora.billing.jwt_auth import (
            extract_user_from_request,
            get_token_blacklist,
        )
        from aragora.server.middleware.auth import extract_token

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # Revoke the current token by adding to blacklist
        token = extract_token(handler)
        if token:
            blacklist = get_token_blacklist()
            if blacklist.revoke_token(token):
                logger.info(f"User logged out and token revoked: {auth_ctx.user_id}")
            else:
                logger.warning(f"User logged out but token revocation failed: {auth_ctx.user_id}")
        else:
            logger.info(f"User logged out (no token to revoke): {auth_ctx.user_id}")

        return json_response({"message": "Logged out successfully"})

    @handle_errors("get user info")
    def _handle_get_me(self, handler) -> HandlerResult:
        """Get current user information."""
        from aragora.billing.jwt_auth import extract_user_from_request

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

    @handle_errors("update user info")
    def _handle_update_me(self, handler) -> HandlerResult:
        """Update current user information."""
        from aragora.billing.jwt_auth import extract_user_from_request

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

    @handle_errors("change password")
    def _handle_change_password(self, handler) -> HandlerResult:
        """Change user password."""
        from aragora.billing.jwt_auth import extract_user_from_request

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

    @handle_errors("revoke token")
    def _handle_revoke_token(self, handler) -> HandlerResult:
        """Explicitly revoke a specific token."""
        from aragora.billing.jwt_auth import (
            extract_user_from_request,
            get_token_blacklist,
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

        # Revoke the token
        blacklist = get_token_blacklist()
        if blacklist.revoke_token(token_to_revoke):
            logger.info(f"Token revoked by user: {auth_ctx.user_id}")
            return json_response({
                "message": "Token revoked successfully",
                "blacklist_size": blacklist.size(),
            })
        else:
            return error_response("Invalid token - could not revoke", 400)

    @handle_errors("generate API key")
    def _handle_generate_api_key(self, handler) -> HandlerResult:
        """Generate a new API key for the user."""
        from aragora.billing.jwt_auth import extract_user_from_request

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
                return error_response(
                    "API access requires Professional tier or higher", 403
                )

        # Generate new API key
        import secrets
        api_key = f"ara_{secrets.token_urlsafe(32)}"
        user_store.update_user(
            user.id,
            api_key=api_key,
            api_key_created_at=datetime.utcnow(),
        )

        logger.info(f"API key generated for user: {user.email}")

        # Return the key (only shown once)
        return json_response(
            {
                "api_key": api_key,
                "message": "Save this key - it will not be shown again",
            }
        )

    @handle_errors("revoke API key")
    def _handle_revoke_api_key(self, handler) -> HandlerResult:
        """Revoke the user's API key."""
        from aragora.billing.jwt_auth import extract_user_from_request

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

        # Revoke API key
        user_store.update_user(user.id, api_key=None, api_key_created_at=None)

        logger.info(f"API key revoked for user: {user.email}")

        return json_response({"message": "API key revoked"})


class InMemoryUserStore:
    """
    Simple in-memory user store for development/testing.

    Production should use a proper database backend.
    """

    def __init__(self):
        self.users: dict[str, Any] = {}  # id -> User
        self.users_by_email: dict[str, str] = {}  # email -> id
        self.organizations: dict[str, Any] = {}  # id -> Organization
        self.api_keys: dict[str, str] = {}  # api_key -> user_id

    def save_user(self, user) -> None:
        """Save a user."""
        self.users[user.id] = user
        self.users_by_email[user.email] = user.id
        if user.api_key:
            self.api_keys[user.api_key] = user.id

    def get_user_by_id(self, user_id: str):
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_by_email(self, email: str):
        """Get user by email."""
        user_id = self.users_by_email.get(email.lower())
        if user_id:
            return self.users.get(user_id)
        return None

    def get_user_by_api_key(self, api_key: str):
        """Get user by API key."""
        user_id = self.api_keys.get(api_key)
        if user_id:
            return self.users.get(user_id)
        return None

    def save_organization(self, org) -> None:
        """Save an organization."""
        self.organizations[org.id] = org

    def get_organization_by_id(self, org_id: str):
        """Get organization by ID."""
        return self.organizations.get(org_id)


__all__ = [
    "AuthHandler",
    "InMemoryUserStore",
    "validate_email",
    "validate_password",
]
