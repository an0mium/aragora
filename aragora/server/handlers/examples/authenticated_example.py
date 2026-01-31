"""
Example: AuthenticatedHandler usage.

This example shows how to use AuthenticatedHandler for handlers that
require user authentication for all endpoints.

Key features demonstrated:
- Automatic authentication verification
- Access to authenticated user context
- Proper error handling for unauthenticated requests
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.protocols import HTTPRequestHandler

from ..base import (
    AuthenticatedHandler,
    HandlerResult,
    json_response,
)

logger = logging.getLogger(__name__)


class ExampleAuthenticatedHandler(AuthenticatedHandler):
    """
    Example handler requiring authentication.

    AuthenticatedHandler extends TypedHandler with:
    - _ensure_authenticated() method for auth verification
    - self.current_user property for accessing user context
    - Automatic 401 responses for unauthenticated requests

    All endpoints in this handler require a valid JWT token.
    """

    ROUTES = [
        "/api/v1/user/profile",
        "/api/v1/user/settings",
        "/api/v1/user/preferences",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """
        Handle GET requests for authenticated user data.

        The _ensure_authenticated() method:
        1. Extracts JWT token from Authorization header
        2. Validates the token
        3. Returns (user_context, None) if valid
        4. Returns (None, error_response) if invalid

        The user context includes:
        - user_id: Unique user identifier
        - email: User's email address
        - org_id: User's organization ID (if applicable)
        - role: User's role (admin, member, etc.)
        - permissions: Set of permission strings
        """
        # First, ensure the user is authenticated
        user, err = self._ensure_authenticated(handler)
        if err:
            # Return 401 Unauthorized
            return err

        # Now we have a guaranteed authenticated user
        # user.user_id, user.email, etc. are available

        if path == "/api/v1/user/profile":
            return self._get_profile(user)

        if path == "/api/v1/user/settings":
            return self._get_settings(user)

        if path == "/api/v1/user/preferences":
            return self._get_preferences(query_params)

        return None

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle POST requests to update user data."""
        # Verify authentication
        user, err = self._ensure_authenticated(handler)
        if err:
            return err

        if path == "/api/v1/user/settings":
            return self._update_settings(handler)

        if path == "/api/v1/user/preferences":
            return self._update_preferences(handler)

        return None

    def _get_profile(self, user: Any) -> HandlerResult:
        """Get user profile information."""
        # Access user properties directly
        return json_response(
            {
                "user_id": user.user_id,
                "email": user.email,
                "org_id": user.org_id,
                "role": getattr(user, "role", "member"),
                "is_authenticated": user.is_authenticated,
            }
        )

    def _get_settings(self, user: Any) -> HandlerResult:
        """Get user settings."""
        # In a real handler, you would fetch from database
        return json_response(
            {
                "user_id": user.user_id,
                "settings": {
                    "notifications_enabled": True,
                    "dark_mode": False,
                    "language": "en",
                },
            }
        )

    def _get_preferences(self, query_params: dict[str, Any]) -> HandlerResult:
        """Get user preferences with optional filtering."""
        # current_user is available via the property after _ensure_authenticated
        user = self.current_user

        category = query_params.get("category", "all")

        return json_response(
            {
                "user_id": user.user_id if user else None,
                "category": category,
                "preferences": {
                    "theme": "system",
                    "timezone": "UTC",
                },
            }
        )

    def _update_settings(self, handler: HTTPRequestHandler) -> HandlerResult:
        """Update user settings."""
        body = self.read_json_body(handler)
        if body is None:
            return self.error_response("Invalid JSON body", 400)

        user = self.current_user

        # In a real handler, you would save to database
        logger.info(f"Updating settings for user {user.user_id}: {body}")

        return json_response(
            {
                "success": True,
                "user_id": user.user_id,
                "updated": list(body.keys()),
            }
        )

    def _update_preferences(self, handler: HTTPRequestHandler) -> HandlerResult:
        """Update user preferences."""
        body = self.read_json_body(handler)
        if body is None:
            return self.error_response("Invalid JSON body", 400)

        user = self.current_user

        return json_response(
            {
                "success": True,
                "user_id": user.user_id,
                "preferences": body,
            }
        )
