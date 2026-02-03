"""
Teams Bot Azure AD OAuth flow handling.

Provides OAuth authentication flow for Microsoft Teams integration:
- Azure AD authentication
- Token verification
- SSO (Single Sign-On) support
- Consent handling
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Environment variables
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID") or os.environ.get("MS_APP_ID")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD")
TEAMS_TENANT_ID = os.environ.get("TEAMS_TENANT_ID")

# Azure AD OAuth endpoints
AZURE_AD_AUTHORITY = "https://login.microsoftonline.com"
BOT_FRAMEWORK_TOKEN_ENDPOINT = "https://login.botframework.com/v1/.well-known/openidconfiguration"


class TeamsOAuth:
    """Handles Azure AD OAuth flow for Teams bot authentication.

    Supports:
    - Bot Framework token validation
    - User SSO token exchange
    - Consent flow handling
    - Token refresh
    """

    def __init__(
        self,
        app_id: str | None = None,
        app_password: str | None = None,
        tenant_id: str | None = None,
    ):
        """Initialize the OAuth handler.

        Args:
            app_id: Microsoft Bot Application ID (defaults to env var).
            app_password: Microsoft Bot Application password (defaults to env var).
            tenant_id: Azure AD Tenant ID for single-tenant apps (defaults to env var).
        """
        self.app_id = app_id or TEAMS_APP_ID or ""
        self.app_password = app_password or TEAMS_APP_PASSWORD or ""
        self.tenant_id = tenant_id or TEAMS_TENANT_ID

        # Token cache (in production, use Redis or similar)
        self._token_cache: dict[str, dict[str, Any]] = {}

    @property
    def authority(self) -> str:
        """Get the Azure AD authority URL."""
        if self.tenant_id:
            return f"{AZURE_AD_AUTHORITY}/{self.tenant_id}"
        return f"{AZURE_AD_AUTHORITY}/common"

    async def get_bot_token(self) -> str | None:
        """Get a Bot Framework access token for outbound API calls.

        Uses client credentials flow to get a token for calling
        Bot Framework APIs (sending messages, etc.).

        Returns:
            Access token string or None if authentication fails.
        """
        try:
            import aiohttp

            token_url = f"{AZURE_AD_AUTHORITY}/botframework.com/oauth2/v2.0/token"

            async with aiohttp.ClientSession() as session:
                data = {
                    "grant_type": "client_credentials",
                    "client_id": self.app_id,
                    "client_secret": self.app_password,
                    "scope": "https://api.botframework.com/.default",
                }

                async with session.post(token_url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("access_token")
                    else:
                        logger.error("Failed to get bot token: %s", response.status)
                        return None

        except ImportError:
            logger.warning("aiohttp not available for OAuth token request")
            return None
        except Exception as e:
            logger.error("Error getting bot token: %s", e)
            return None

    async def validate_incoming_token(self, auth_header: str) -> dict[str, Any] | None:
        """Validate an incoming Bot Framework JWT token.

        Delegates to the centralized JWT verifier for signature and claims
        validation.

        Args:
            auth_header: The full Authorization header value.

        Returns:
            Decoded token claims if valid, None otherwise.
        """
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix

        try:
            from aragora.connectors.chat.jwt_verify import HAS_JWT, decode_teams_token  # type: ignore[attr-defined]

            if HAS_JWT:
                return decode_teams_token(token, self.app_id)
            else:
                logger.warning("JWT verification not available")
                return None

        except ImportError:
            logger.warning("JWT verification module not available")
            return None
        except Exception as e:
            logger.error("Token validation error: %s", e)
            return None

    async def exchange_sso_token(self, sso_token: str, connection_name: str = "") -> str | None:
        """Exchange a Teams SSO token for a user access token.

        When a user interacts with the bot, Teams can provide an SSO token
        that can be exchanged for an access token to call Graph API or
        other services on behalf of the user.

        Args:
            sso_token: The SSO token from Teams activity.
            connection_name: OAuth connection name configured in Bot Framework.

        Returns:
            User access token or None if exchange fails.
        """
        try:
            # Use Bot Framework token exchange
            # This requires the bot to have an OAuth connection configured
            logger.debug("Attempting SSO token exchange")

            # In production, this would call the Bot Framework Token Service
            # to exchange the SSO token for a user access token
            #
            # token_exchange_url = (
            #     f"https://api.botframework.com/api/usertoken/token"
            #     f"?userId={user_id}&connectionName={connection_name}"
            # )

            # For now, return None as this requires additional Bot Framework setup
            logger.debug("SSO token exchange not implemented - requires OAuth connection")
            return None

        except Exception as e:
            logger.error("SSO token exchange failed: %s", e)
            return None

    async def get_user_token(self, user_id: str, connection_name: str) -> dict[str, Any] | None:
        """Get a cached user token from the Bot Framework Token Service.

        Args:
            user_id: The user's ID from the activity.
            connection_name: OAuth connection name configured in Bot Framework.

        Returns:
            Token response dict or None if not cached.
        """
        cache_key = f"{user_id}:{connection_name}"
        return self._token_cache.get(cache_key)

    async def sign_out_user(self, user_id: str, connection_name: str) -> bool:
        """Sign out a user by clearing their cached tokens.

        Args:
            user_id: The user's ID.
            connection_name: OAuth connection name.

        Returns:
            True if sign out was successful.
        """
        cache_key = f"{user_id}:{connection_name}"
        if cache_key in self._token_cache:
            del self._token_cache[cache_key]
            logger.info("Signed out user %s from %s", user_id, connection_name)
            return True
        return False

    def build_oauth_card(
        self, connection_name: str, sign_in_text: str = "Sign in"
    ) -> dict[str, Any]:
        """Build an OAuth card for user authentication.

        Returns an Adaptive Card that triggers the Teams OAuth flow
        when the user clicks the sign-in button.

        Args:
            connection_name: OAuth connection name configured in Bot Framework.
            sign_in_text: Text to display on the sign-in button.

        Returns:
            Adaptive Card dict.
        """
        return {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": "Sign In Required",
                    "weight": "Bolder",
                    "size": "Medium",
                },
                {
                    "type": "TextBlock",
                    "text": "Please sign in to continue using Aragora features.",
                    "wrap": True,
                },
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": sign_in_text,
                    "data": {
                        "action": "oauth_signin",
                        "connection_name": connection_name,
                    },
                },
            ],
        }

    def validate_tenant(
        self, activity_tenant_id: str, expected_tenant_id: str | None = None
    ) -> bool:
        """Validate that an activity comes from an expected tenant.

        For multi-tenant scenarios, validates the request origin.

        Args:
            activity_tenant_id: Tenant ID from the activity.
            expected_tenant_id: Expected tenant ID (defaults to configured tenant).

        Returns:
            True if the tenant is valid.
        """
        required_tenant = expected_tenant_id or self.tenant_id

        if not required_tenant:
            # No tenant restriction configured
            return True

        return activity_tenant_id == required_tenant


__all__ = [
    "TeamsOAuth",
    "AZURE_AD_AUTHORITY",
    "BOT_FRAMEWORK_TOKEN_ENDPOINT",
]
