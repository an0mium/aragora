"""
Microsoft Teams OAuth Handler for app installation flow.

Endpoints:
- GET  /api/integrations/teams/install   - Redirect to Microsoft OAuth
- GET  /api/integrations/teams/callback  - Handle OAuth callback
- POST /api/integrations/teams/refresh   - Refresh expired tokens

Environment Variables:
- TEAMS_CLIENT_ID: Azure AD application (client) ID
- TEAMS_CLIENT_SECRET: Azure AD client secret
- TEAMS_REDIRECT_URI: OAuth callback URL (defaults to /api/integrations/teams/callback)

Microsoft OAuth URLs:
- Authorize: https://login.microsoftonline.com/common/oauth2/v2.0/authorize
- Token: https://login.microsoftonline.com/common/oauth2/v2.0/token
- Scopes: https://graph.microsoft.com/.default offline_access

See: https://learn.microsoft.com/en-us/azure/active-directory/develop/v2-oauth2-auth-code-flow
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
TEAMS_CLIENT_ID = os.environ.get("TEAMS_CLIENT_ID", "")
TEAMS_CLIENT_SECRET = os.environ.get("TEAMS_CLIENT_SECRET", "")
TEAMS_REDIRECT_URI = os.environ.get("TEAMS_REDIRECT_URI", "")

# Default OAuth scopes for Aragora Teams app
DEFAULT_SCOPES = "https://graph.microsoft.com/.default offline_access"
TEAMS_SCOPES = os.environ.get("TEAMS_SCOPES", DEFAULT_SCOPES)

# Microsoft OAuth URLs
MS_OAUTH_AUTHORIZE_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
MS_OAUTH_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"

# State token storage (in production, use Redis or database)
_oauth_states: Dict[str, Dict[str, Any]] = {}


class TeamsOAuthHandler(BaseHandler):
    """Handler for Microsoft Teams OAuth installation flow."""

    ROUTES = [
        "/api/integrations/teams/install",
        "/api/integrations/teams/callback",
        "/api/integrations/teams/refresh",
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

        if path == "/api/integrations/teams/install":
            if method == "GET":
                return await self._handle_install(query_params)
            return error_response("Method not allowed", 405)

        elif path == "/api/integrations/teams/callback":
            if method == "GET":
                return await self._handle_callback(query_params)
            return error_response("Method not allowed", 405)

        elif path == "/api/integrations/teams/refresh":
            if method == "POST":
                return await self._handle_refresh(body)
            return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    async def _handle_install(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Initiate Microsoft Teams OAuth installation flow.

        Redirects user to Microsoft's OAuth consent page.
        Optional query params:
            org_id: Aragora organization to link tenant to
        """
        if not TEAMS_CLIENT_ID:
            return error_response(
                "Teams OAuth not configured. Set TEAMS_CLIENT_ID environment variable.",
                503,
            )

        # Generate state token for CSRF protection
        state = secrets.token_urlsafe(32)
        org_id = query_params.get("org_id")

        # Store state with metadata
        _oauth_states[state] = {
            "created_at": time.time(),
            "org_id": org_id,
        }

        # Clean up old states (older than 10 minutes)
        current_time = time.time()
        expired_states = [
            s for s, data in _oauth_states.items() if current_time - data["created_at"] > 600
        ]
        for s in expired_states:
            del _oauth_states[s]

        # Build OAuth URL
        redirect_uri = TEAMS_REDIRECT_URI
        if not redirect_uri:
            # Try to construct from request
            host = query_params.get("host", "localhost:8080")
            scheme = "https" if "localhost" not in host else "http"
            redirect_uri = f"{scheme}://{host}/api/integrations/teams/callback"

        oauth_params = {
            "client_id": TEAMS_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "response_mode": "query",
            "scope": TEAMS_SCOPES,
            "state": state,
        }

        oauth_url = f"{MS_OAUTH_AUTHORIZE_URL}?{urlencode(oauth_params)}"

        logger.info(f"Initiating Teams OAuth flow (state: {state[:8]}...)")

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
        Handle OAuth callback from Microsoft.

        Query params:
            code: Authorization code from Microsoft
            state: State token for CSRF verification
            error: Error code if user denied
            error_description: Error details
        """
        # Check for error from Microsoft
        if "error" in query_params:
            error_code = query_params.get("error")
            error_desc = query_params.get("error_description", "")
            logger.warning(f"Teams OAuth error: {error_code} - {error_desc}")
            return error_response(f"Microsoft authorization denied: {error_code}", 400)

        code = query_params.get("code")
        state = query_params.get("state")

        if not code:
            return error_response("Missing authorization code", 400)

        if not state:
            return error_response("Missing state parameter", 400)

        # Verify state token
        state_data = _oauth_states.pop(state, None)
        if not state_data:
            return error_response("Invalid or expired state token", 400)

        org_id = state_data.get("org_id")

        if not TEAMS_CLIENT_ID or not TEAMS_CLIENT_SECRET:
            return error_response("Teams OAuth not configured", 503)

        # Exchange code for access token
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    MS_OAUTH_TOKEN_URL,
                    data={
                        "client_id": TEAMS_CLIENT_ID,
                        "client_secret": TEAMS_CLIENT_SECRET,
                        "code": code,
                        "redirect_uri": TEAMS_REDIRECT_URI,
                        "grant_type": "authorization_code",
                        "scope": TEAMS_SCOPES,
                    },
                )
                response.raise_for_status()
                data = response.json()

        except ImportError:
            return error_response("httpx not available", 503)
        except httpx.HTTPStatusError as e:
            logger.error(f"Teams token exchange failed: {e.response.text}")
            return error_response(f"Token exchange failed: {e}", 500)
        except Exception as e:
            logger.error(f"Teams token exchange failed: {e}")
            return error_response(f"Token exchange failed: {e}", 500)

        # Extract token info
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in", 3600)
        scope = data.get("scope", "")

        if not access_token:
            return error_response("Invalid response from Microsoft", 500)

        # Get tenant info from Microsoft Graph
        tenant_id = ""
        tenant_name = "Unknown"
        bot_id = ""

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get organization info
                org_response = await client.get(
                    "https://graph.microsoft.com/v1.0/organization",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if org_response.status_code == 200:
                    org_data = org_response.json()
                    if org_data.get("value"):
                        org_info = org_data["value"][0]
                        tenant_id = org_info.get("id", "")
                        tenant_name = org_info.get("displayName", "Unknown")

                # Get bot/application info
                me_response = await client.get(
                    "https://graph.microsoft.com/v1.0/me",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if me_response.status_code == 200:
                    me_data = me_response.json()
                    bot_id = me_data.get("id", TEAMS_CLIENT_ID)

        except Exception as e:
            logger.warning(f"Failed to fetch tenant info: {e}")
            tenant_id = tenant_id or "unknown"
            bot_id = bot_id or TEAMS_CLIENT_ID

        if not tenant_id:
            return error_response("Could not determine tenant ID", 500)

        # Calculate token expiration
        expires_at = time.time() + expires_in

        # Store tenant credentials
        try:
            from aragora.storage.teams_tenant_store import (
                TeamsTenant,
                get_teams_tenant_store,
            )

            tenant = TeamsTenant(
                tenant_id=tenant_id,
                tenant_name=tenant_name,
                access_token=access_token,
                refresh_token=refresh_token,
                bot_id=bot_id,
                installed_at=time.time(),
                installed_by=None,  # Could extract from token if needed
                scopes=scope.split(" ") if scope else [],
                aragora_org_id=org_id,
                is_active=True,
                expires_at=expires_at,
            )

            store = get_teams_tenant_store()
            if not store.save(tenant):
                return error_response("Failed to save tenant", 500)

            logger.info(f"Teams tenant installed: {tenant_name} ({tenant_id})")

        except ImportError as e:
            logger.error(f"Tenant store not available: {e}")
            return error_response("Tenant storage not available", 503)

        # Return success page
        success_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Aragora - Teams Connected</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #6264A7 0%, #464775 100%);
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
                .tenant {{
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
                <p>Aragora is now installed in</p>
                <p class="tenant">{tenant_name}</p>
                <p>You can close this window and return to Teams.</p>
            </div>
        </body>
        </html>
        """

        return HandlerResult(
            status_code=200,
            content_type="text/html",
            body=success_html.encode("utf-8"),
        )

    async def _handle_refresh(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Refresh expired tokens for a tenant.

        Request body:
            tenant_id: Azure AD tenant ID to refresh
        """
        tenant_id = body.get("tenant_id")
        if not tenant_id:
            return error_response("Missing tenant_id", 400)

        if not TEAMS_CLIENT_ID or not TEAMS_CLIENT_SECRET:
            return error_response("Teams OAuth not configured", 503)

        # Get tenant from store
        try:
            from aragora.storage.teams_tenant_store import get_teams_tenant_store

            store = get_teams_tenant_store()
            tenant = store.get(tenant_id)

            if not tenant:
                return error_response("Tenant not found", 404)

            if not tenant.refresh_token:
                return error_response("No refresh token available", 400)

        except ImportError:
            return error_response("Tenant storage not available", 503)

        # Refresh the token
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    MS_OAUTH_TOKEN_URL,
                    data={
                        "client_id": TEAMS_CLIENT_ID,
                        "client_secret": TEAMS_CLIENT_SECRET,
                        "refresh_token": tenant.refresh_token,
                        "grant_type": "refresh_token",
                        "scope": TEAMS_SCOPES,
                    },
                )
                response.raise_for_status()
                data = response.json()

        except ImportError:
            return error_response("httpx not available", 503)
        except httpx.HTTPStatusError as e:
            logger.error(f"Teams token refresh failed: {e.response.text}")
            return error_response(f"Token refresh failed: {e}", 500)
        except Exception as e:
            logger.error(f"Teams token refresh failed: {e}")
            return error_response(f"Token refresh failed: {e}", 500)

        # Extract new token info
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token", tenant.refresh_token)  # May not be returned
        expires_in = data.get("expires_in", 3600)

        if not access_token:
            return error_response("Invalid response from Microsoft", 500)

        # Calculate new expiration
        expires_at = time.time() + expires_in

        # Update tokens in store
        if not store.update_tokens(tenant_id, access_token, refresh_token, expires_at):
            return error_response("Failed to update tokens", 500)

        logger.info(f"Teams tokens refreshed for tenant: {tenant_id}")

        return json_response(
            {
                "success": True,
                "tenant_id": tenant_id,
                "expires_in": expires_in,
            }
        )


# Handler factory function for registration
def create_teams_oauth_handler(server_context: Any) -> TeamsOAuthHandler:
    """Factory function for handler registration."""
    return TeamsOAuthHandler(server_context)
