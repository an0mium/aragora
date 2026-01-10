"""
User Authentication Handlers.

Endpoints:
- POST /api/auth/register - Create a new user account
- POST /api/auth/login - Authenticate and get tokens
- POST /api/auth/logout - Invalidate current token
- POST /api/auth/refresh - Refresh access token
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

        if path == "/api/auth/api-key":
            if method == "POST":
                return self._handle_generate_api_key(handler)
            elif method == "DELETE":
                return self._handle_revoke_api_key(handler)

        return error_response("Method not allowed", 405)

    def _get_user_store(self):
        """Get user store from context."""
        return self.ctx.get("user_store")

    @handle_errors("user registration")
    @log_request("user registration")
    def _handle_register(self, handler) -> HandlerResult:
        """Handle user registration."""
        from aragora.billing.models import User, Organization, generate_slug
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
            # In-memory fallback for development
            user_store = self.ctx.setdefault("user_store", InMemoryUserStore())

        # Check if email already exists
        existing = user_store.get_user_by_email(email)
        if existing:
            return error_response("Email already registered", 409)

        # Create organization if name provided
        org_id = None
        if org_name:
            org = Organization(
                name=org_name,
                slug=generate_slug(org_name),
            )
            user_store.save_organization(org)
            org_id = org.id

        # Create user
        user = User(
            email=email,
            name=name or email.split("@")[0],
            org_id=org_id,
            role="owner" if org_id else "member",
        )
        user.set_password(password)

        # If org created, set user as owner
        if org_id:
            org.owner_id = user.id
            user_store.save_organization(org)

        # Save user
        user_store.save_user(user)

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
        user.last_login_at = datetime.utcnow()
        user_store.save_user(user)

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
        from aragora.billing.jwt_auth import extract_user_from_request

        # Get current user
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)

        # In a production system, we would add the token to a blacklist
        # For now, just acknowledge the logout
        logger.info(f"User logged out: {auth_ctx.user_id}")

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
        if "name" in body:
            user.name = str(body["name"]).strip()[:100]

        # Save updates
        user.updated_at = datetime.utcnow()
        user_store.save_user(user)

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
        user.set_password(new_password)
        user_store.save_user(user)

        logger.info(f"Password changed for user: {user.email}")

        return json_response({"message": "Password changed successfully"})

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
        api_key = user.generate_api_key()
        user_store.save_user(user)

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
        user.revoke_api_key()
        user_store.save_user(user)

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
