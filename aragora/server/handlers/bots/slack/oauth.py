"""
Slack OAuth Flow Handler.

This module handles OAuth2 installation flow for Slack apps including:
- OAuth authorization redirect
- OAuth callback handling
- Token storage and management
"""

import json
import logging
import os
from typing import Any

from aragora.audit.unified import audit_data
from aragora.server.handlers.base import HandlerResult, error_response, json_response

from .constants import _validate_slack_team_id

logger = logging.getLogger(__name__)


async def handle_slack_oauth_start(request: Any) -> HandlerResult:
    """Start the Slack OAuth installation flow.

    Redirects user to Slack's OAuth authorization page with proper scopes.
    """
    client_id = os.environ.get("SLACK_CLIENT_ID", "")
    redirect_uri = os.environ.get("SLACK_REDIRECT_URI", "")

    if not client_id:
        logger.error("SLACK_CLIENT_ID not configured")
        return error_response("Slack OAuth not configured", 503)

    # Required scopes for the bot
    scopes = [
        "app_mentions:read",
        "channels:history",
        "channels:read",
        "chat:write",
        "commands",
        "groups:history",
        "groups:read",
        "im:history",
        "im:read",
        "im:write",
        "reactions:read",
        "reactions:write",
        "users:read",
    ]

    oauth_url = (
        f"https://slack.com/oauth/v2/authorize"
        f"?client_id={client_id}"
        f"&scope={','.join(scopes)}"
        f"&redirect_uri={redirect_uri}"
    )

    # Return redirect response
    return json_response(
        {"redirect_url": oauth_url},
        status=302,
        headers={"Location": oauth_url},
    )


async def handle_slack_oauth_callback(request: Any) -> HandlerResult:
    """Handle the OAuth callback from Slack.

    Exchanges the authorization code for access tokens and stores them.
    """
    try:
        # Get query parameters
        if hasattr(request, "query_params"):
            params = dict(request.query_params)
        elif hasattr(request, "args"):
            params = dict(request.args)
        else:
            params = {}

        code = params.get("code", "")
        error = params.get("error", "")

        if error:
            logger.warning("Slack OAuth error: %s", error)
            return error_response(f"Slack OAuth error: {error}", 400)

        if not code:
            logger.warning("No authorization code in OAuth callback")
            return error_response("Missing authorization code", 400)

        # Exchange code for tokens
        client_id = os.environ.get("SLACK_CLIENT_ID", "")
        client_secret = os.environ.get("SLACK_CLIENT_SECRET", "")
        redirect_uri = os.environ.get("SLACK_REDIRECT_URI", "")

        if not client_id or not client_secret:
            logger.error("Slack OAuth credentials not configured")
            return error_response("Slack OAuth not configured", 503)

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://slack.com/api/oauth.v2.access",
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "code": code,
                        "redirect_uri": redirect_uri,
                    },
                )
                data = response.json()

        except ImportError:
            logger.error("httpx not available for OAuth exchange")
            return error_response("OAuth exchange failed: missing dependencies", 500)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.error("Failed to exchange OAuth code: %s", e)
            return error_response("OAuth exchange failed", 500)

        if not data.get("ok"):
            error_msg = data.get("error", "unknown_error")
            logger.error("Slack OAuth exchange failed: %s", error_msg)
            return error_response(f"Slack OAuth failed: {error_msg}", 400)

        # Extract tokens and team info
        team = data.get("team", {})
        team_id = team.get("id", "")
        team_name = team.get("name", "")

        # Validate team_id
        valid, error = _validate_slack_team_id(team_id)
        if not valid:
            logger.error("Invalid team_id from OAuth: %s", error)
            return error_response(f"Invalid team ID: {error}", 400)

        access_token = data.get("access_token", "")
        bot_user_id = data.get("bot_user_id", "")

        # Store the tokens
        try:
            from aragora.storage.slack_workspace_store import get_slack_workspace_store

            store = get_slack_workspace_store()
            store.store_workspace(
                team_id=team_id,
                team_name=team_name,
                access_token=access_token,
                bot_user_id=bot_user_id,
                authed_user=data.get("authed_user", {}),
            )
            logger.info("Stored Slack workspace credentials for %s (%s)", team_name, team_id)

            audit_data(
                user_id="system",
                resource_type="slack_workspace",
                resource_id=team_id,
                action="install",
                platform="slack",
                team_name=team_name,
            )

        except (ImportError, RuntimeError, OSError) as e:
            logger.error("Failed to store workspace credentials: %s", e)
            return error_response("Failed to complete installation", 500)

        return json_response(
            {
                "success": True,
                "team_id": team_id,
                "team_name": team_name,
                "message": f"Aragora successfully installed to {team_name}!",
            }
        )

    except (TypeError, ValueError, KeyError, AttributeError, OSError) as e:
        logger.exception("Unexpected error in OAuth callback: %s", e)
        return error_response("OAuth callback failed", 500)


async def handle_slack_oauth_revoke(request: Any) -> HandlerResult:
    """Handle OAuth token revocation.

    Revokes the access token for a Slack workspace.
    """
    try:
        body = await request.body()
        import json

        data = json.loads(body) if body else {}

        team_id = data.get("team_id", "")

        # Validate team_id
        valid, error = _validate_slack_team_id(team_id)
        if not valid:
            logger.warning("Invalid team_id in revoke request: %s", error)
            return error_response(error or "Invalid team ID", 400)

        try:
            from aragora.storage.slack_workspace_store import get_slack_workspace_store

            store = get_slack_workspace_store()
            store.revoke_token(team_id)
            logger.info("Revoked Slack tokens for workspace %s", team_id)

            audit_data(
                user_id="system",
                resource_type="slack_workspace",
                resource_id=team_id,
                action="revoke",
                platform="slack",
            )

        except (ImportError, RuntimeError, OSError) as e:
            logger.error("Failed to revoke workspace tokens: %s", e)
            return error_response("Failed to revoke tokens", 500)

        return json_response({"success": True, "team_id": team_id})

    except (TypeError, ValueError, KeyError, AttributeError, json.JSONDecodeError) as e:
        logger.exception("Unexpected error in revoke handler: %s", e)
        return error_response("Revocation failed", 500)


__all__ = [
    "handle_slack_oauth_start",
    "handle_slack_oauth_callback",
    "handle_slack_oauth_revoke",
]
