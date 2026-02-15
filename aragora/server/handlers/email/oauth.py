"""
Gmail OAuth handlers.

Provides handlers for:
- OAuth URL generation
- OAuth callback handling
- Gmail connection status
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.server.middleware.rate_limit import rate_limit
from aragora.observability.metrics import track_handler

from .storage import _check_email_permission, get_gmail_connector

logger = logging.getLogger(__name__)


@rate_limit(requests_per_minute=30)
@track_handler("email/gmail/oauth/url")
async def handle_gmail_oauth_url(
    redirect_uri: str,
    state: str = "",
    scopes: str = "readonly",  # "readonly" or "full"
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Get Gmail OAuth authorization URL.

    POST /api/email/gmail/oauth/url
    {
        "redirect_uri": "https://app.example.com/oauth/callback",
        "state": "user_123",
        "scopes": "full"
    }
    """
    # Check RBAC permission
    perm_error = _check_email_permission(auth_context, "email:oauth")
    if perm_error:
        return perm_error

    try:
        connector = get_gmail_connector()

        # Set scopes based on request
        if scopes == "full":
            from aragora.connectors.enterprise.communication.gmail import GMAIL_SCOPES_FULL

            connector._scopes = GMAIL_SCOPES_FULL
        else:
            from aragora.connectors.enterprise.communication.gmail import GMAIL_SCOPES_READONLY

            connector._scopes = GMAIL_SCOPES_READONLY

        url = connector.get_oauth_url(redirect_uri, state)

        return {
            "success": True,
            "oauth_url": url,
            "scopes": scopes,
        }

    except Exception as e:
        logger.exception(f"Failed to get OAuth URL: {e}")
        return {
            "success": False,
            "error": "Failed to get OAuth URL",
        }


@rate_limit(requests_per_minute=5)  # Strict rate limit for OAuth callback security
@track_handler("email/gmail/oauth/callback")
async def handle_gmail_oauth_callback(
    code: str,
    redirect_uri: str,
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Handle Gmail OAuth callback and store tokens.

    POST /api/email/gmail/oauth/callback
    {
        "code": "auth_code_from_google",
        "redirect_uri": "https://app.example.com/oauth/callback"
    }
    """
    # Check RBAC permission
    perm_error = _check_email_permission(auth_context, "email:oauth")
    if perm_error:
        return perm_error

    try:
        connector = get_gmail_connector(user_id)
        await connector.authenticate(code=code, redirect_uri=redirect_uri)

        # Get user info to confirm
        user_info = await connector.get_user_info()

        return {
            "success": True,
            "authenticated": True,
            "email": user_info.get("emailAddress"),
            "messages_total": user_info.get("messagesTotal"),
        }

    except Exception as e:
        logger.exception(f"OAuth callback failed: {e}")
        return {
            "success": False,
            "error": "OAuth callback failed",
        }


@rate_limit(requests_per_minute=60)
@track_handler("email/gmail/status", method="GET")
async def handle_gmail_status(
    user_id: str = "default",
    workspace_id: str = "default",
    auth_context: Any | None = None,
) -> dict[str, Any]:
    """
    Check Gmail connection status.

    GET /api/email/gmail/status
    """
    # Check RBAC permission
    perm_error = _check_email_permission(auth_context, "email:read")
    if perm_error:
        return perm_error

    try:
        connector = get_gmail_connector(user_id)
        is_authenticated = connector._access_token is not None

        result: dict[str, Any] = {
            "success": True,
            "authenticated": is_authenticated,
        }

        if is_authenticated:
            try:
                user_info = await connector.get_user_info()
                result["email"] = user_info.get("emailAddress")
                result["messages_total"] = user_info.get("messagesTotal")
            except (KeyError, AttributeError) as e:
                logger.debug(f"Failed to extract user info fields: {e}")
                result["authenticated"] = False
                result["error"] = "Token expired or invalid"
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Network error checking Gmail status: {e}")
                result["authenticated"] = False
                result["error"] = "Token expired or invalid"
            except Exception as e:
                logger.warning(f"Unexpected error checking Gmail status: {e}")
                result["authenticated"] = False
                result["error"] = "Token expired or invalid"

        return result

    except Exception as e:
        logger.exception(f"Failed to check Gmail status: {e}")
        return {
            "success": False,
            "error": "Failed to check Gmail status",
        }
