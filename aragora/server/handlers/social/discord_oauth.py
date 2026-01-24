"""
Discord OAuth Handler for bot installation flow.

Endpoints:
- GET  /api/integrations/discord/install   - Redirect to Discord OAuth
- GET  /api/integrations/discord/callback  - Handle OAuth callback
- POST /api/integrations/discord/uninstall - Handle guild removal

Environment Variables:
- DISCORD_CLIENT_ID: Discord application client ID
- DISCORD_CLIENT_SECRET: Discord application client secret
- DISCORD_REDIRECT_URI: OAuth callback URL (defaults to /api/integrations/discord/callback)

Discord OAuth URLs:
- Authorize: https://discord.com/api/oauth2/authorize
- Token: https://discord.com/api/oauth2/token
- Scopes: bot applications.commands

See: https://discord.com/developers/docs/topics/oauth2
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

# Environment configuration
DISCORD_CLIENT_ID = os.environ.get("DISCORD_CLIENT_ID", "")
DISCORD_CLIENT_SECRET = os.environ.get("DISCORD_CLIENT_SECRET", "")
DISCORD_REDIRECT_URI = os.environ.get("DISCORD_REDIRECT_URI", "")

# Default OAuth scopes for Aragora Discord bot
DEFAULT_SCOPES = "bot applications.commands"
DISCORD_SCOPES = os.environ.get("DISCORD_SCOPES", DEFAULT_SCOPES)

# Default bot permissions (send messages, read messages, embed links, add reactions)
DEFAULT_PERMISSIONS = "274877975616"
DISCORD_BOT_PERMISSIONS = os.environ.get("DISCORD_BOT_PERMISSIONS", DEFAULT_PERMISSIONS)

# Discord OAuth URLs
DISCORD_OAUTH_AUTHORIZE_URL = "https://discord.com/api/oauth2/authorize"
DISCORD_OAUTH_TOKEN_URL = "https://discord.com/api/oauth2/token"
DISCORD_API_BASE = "https://discord.com/api/v10"

# State token storage (in production, use Redis or database)
_oauth_states: Dict[str, Dict[str, Any]] = {}


class DiscordOAuthHandler(BaseHandler):
    """Handler for Discord OAuth bot installation flow."""

    ROUTES = [
        "/api/integrations/discord/install",
        "/api/integrations/discord/callback",
        "/api/integrations/discord/uninstall",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    async def handle(  # type: ignore[override]
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HandlerResult:
        """Route OAuth requests to appropriate methods."""
        query_params = query_params or {}
        body = body or {}

        if path == "/api/integrations/discord/install":
            if method == "GET":
                return await self._handle_install(query_params)
            return error_response("Method not allowed", 405)

        elif path == "/api/integrations/discord/callback":
            if method == "GET":
                return await self._handle_callback(query_params)
            return error_response("Method not allowed", 405)

        elif path == "/api/integrations/discord/uninstall":
            if method == "POST":
                return await self._handle_uninstall(body)
            return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    async def _handle_install(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Initiate Discord OAuth bot installation flow.

        Redirects user to Discord's OAuth consent page.
        Optional query params:
            tenant_id: Aragora tenant to link guild to
        """
        if not DISCORD_CLIENT_ID:
            return error_response(
                "Discord OAuth not configured. Set DISCORD_CLIENT_ID environment variable.",
                503,
            )

        # Generate state token for CSRF protection
        state = secrets.token_urlsafe(32)
        tenant_id = query_params.get("tenant_id")

        # Store state with metadata
        _oauth_states[state] = {
            "created_at": time.time(),
            "tenant_id": tenant_id,
        }

        # Clean up old states (older than 10 minutes)
        current_time = time.time()
        expired_states = [
            s for s, data in _oauth_states.items() if current_time - data["created_at"] > 600
        ]
        for s in expired_states:
            del _oauth_states[s]

        # Build OAuth URL
        redirect_uri = DISCORD_REDIRECT_URI
        if not redirect_uri:
            # Try to construct from request
            host = query_params.get("host", "localhost:8080")
            scheme = "https" if "localhost" not in host else "http"
            redirect_uri = f"{scheme}://{host}/api/integrations/discord/callback"

        oauth_params = {
            "client_id": DISCORD_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": DISCORD_SCOPES,
            "permissions": DISCORD_BOT_PERMISSIONS,
            "state": state,
        }

        oauth_url = f"{DISCORD_OAUTH_AUTHORIZE_URL}?{urlencode(oauth_params)}"

        logger.info(f"Initiating Discord OAuth flow (state: {state[:8]}...)")

        # Return redirect response
        return HandlerResult(
            status_code=302,
            content_type="text/html",
            body=b"",
            headers={
                "Location": oauth_url,
                "Cache-Control": "no-store",
            },
        )

    async def _handle_callback(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Handle OAuth callback from Discord.

        Query params:
            code: Authorization code from Discord
            state: State token for CSRF verification
            guild_id: ID of guild where bot was added
            error: Error code if user denied
            error_description: Error details
        """
        # Check for error from Discord
        if "error" in query_params:
            error_code = query_params.get("error")
            error_desc = query_params.get("error_description", "")
            logger.warning(f"Discord OAuth error: {error_code} - {error_desc}")
            return error_response(f"Discord authorization denied: {error_code}", 400)

        code = query_params.get("code")
        state = query_params.get("state")
        guild_id = query_params.get("guild_id", "")

        if not code:
            return error_response("Missing authorization code", 400)

        if not state:
            return error_response("Missing state parameter", 400)

        # Verify state token
        state_data = _oauth_states.pop(state, None)
        if not state_data:
            return error_response("Invalid or expired state token", 400)

        tenant_id = state_data.get("tenant_id")

        if not DISCORD_CLIENT_ID or not DISCORD_CLIENT_SECRET:
            return error_response("Discord OAuth not configured", 503)

        # Build redirect URI (same as install)
        redirect_uri = DISCORD_REDIRECT_URI
        if not redirect_uri:
            host = query_params.get("host", "localhost:8080")
            scheme = "https" if "localhost" not in host else "http"
            redirect_uri = f"{scheme}://{host}/api/integrations/discord/callback"

        # Exchange code for access token
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    DISCORD_OAUTH_TOKEN_URL,
                    data={
                        "client_id": DISCORD_CLIENT_ID,
                        "client_secret": DISCORD_CLIENT_SECRET,
                        "code": code,
                        "redirect_uri": redirect_uri,
                        "grant_type": "authorization_code",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                data = response.json()

        except ImportError:
            return error_response("httpx not available", 503)
        except Exception as e:
            logger.error(f"Discord token exchange failed: {e}")
            return error_response(f"Token exchange failed: {e}", 500)

        # Extract token info
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in", 604800)  # Default 7 days
        scope = data.get("scope", "")

        if not access_token:
            return error_response("Invalid response from Discord", 500)

        # Get guild info from Discord
        guild_name = "Unknown"
        bot_user_id = ""

        # For bot flow, guild info is provided in the response
        guild_data = data.get("guild", {})
        if guild_data:
            guild_id = guild_data.get("id", guild_id)
            guild_name = guild_data.get("name", "Unknown")

        # Get bot user info
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get current bot user
                me_response = await client.get(
                    f"{DISCORD_API_BASE}/users/@me",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if me_response.status_code == 200:
                    me_data = me_response.json()
                    bot_user_id = me_data.get("id", "")

        except Exception as e:
            logger.warning(f"Failed to fetch bot info: {e}")
            bot_user_id = bot_user_id or DISCORD_CLIENT_ID

        if not guild_id:
            return error_response("Could not determine guild ID", 500)

        # Calculate token expiration
        expires_at = time.time() + expires_in

        # Store guild credentials
        try:
            from aragora.storage.discord_guild_store import (
                DiscordGuild,
                get_discord_guild_store,
            )

            guild = DiscordGuild(
                guild_id=guild_id,
                guild_name=guild_name,
                access_token=access_token,
                refresh_token=refresh_token,
                bot_user_id=bot_user_id,
                installed_at=time.time(),
                installed_by=None,  # Could extract from token if needed
                scopes=scope.split(" ") if scope else [],
                tenant_id=tenant_id,
                is_active=True,
                expires_at=expires_at,
            )

            store = get_discord_guild_store()
            if not store.save(guild):
                return error_response("Failed to save guild", 500)

            logger.info(f"Discord guild installed: {guild_name} ({guild_id})")

        except ImportError as e:
            logger.error(f"Guild store not available: {e}")
            return error_response("Guild storage not available", 503)

        # Return success page
        success_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Aragora - Discord Connected</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #5865F2 0%, #7289da 100%);
                }}
                .card {{
                    background: white;
                    padding: 2rem 3rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                    text-align: center;
                    max-width: 400px;
                }}
                h1 {{ color: #2d3748; margin-bottom: 0.5rem; }}
                p {{ color: #718096; }}
                .guild {{
                    font-weight: bold;
                    color: #4a5568;
                    font-size: 1.1rem;
                }}
                .check {{
                    font-size: 3rem;
                    color: #48bb78;
                    margin-bottom: 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="card">
                <div class="check">&#10003;</div>
                <h1>Connected!</h1>
                <p>Aragora bot has been added to</p>
                <p class="guild">{guild_name}</p>
                <p>You can close this window and return to Discord.</p>
            </div>
        </body>
        </html>
        """

        return HandlerResult(
            status_code=200,
            content_type="text/html",
            body=success_html.encode("utf-8"),
        )

    async def _handle_uninstall(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Handle bot removal webhook from Discord.

        Request body:
            guild_id: Discord guild ID where bot was removed
        """
        guild_id = body.get("guild_id")

        if not guild_id:
            return error_response("Missing guild_id", 400)

        try:
            from aragora.storage.discord_guild_store import get_discord_guild_store

            store = get_discord_guild_store()
            store.deactivate(guild_id)
            logger.info(f"Discord guild deactivated: {guild_id}")

        except ImportError:
            logger.warning("Could not deactivate guild - store unavailable")

        return json_response({"ok": True, "guild_id": guild_id})


# Handler factory function for registration
def create_discord_oauth_handler(server_context: Any) -> DiscordOAuthHandler:
    """Factory function for handler registration."""
    return DiscordOAuthHandler(server_context)
