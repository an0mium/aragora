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
import time
from typing import Any
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)
from ..secure import ForbiddenError, SecureHandler, UnauthorizedError

# RBAC Permission constants for Teams OAuth
# Following granular permission model for OAuth security
PERM_TEAMS_OAUTH_INSTALL = "teams:oauth:install"
PERM_TEAMS_OAUTH_CALLBACK = "teams:oauth:callback"
PERM_TEAMS_OAUTH_DISCONNECT = "teams:oauth:disconnect"
PERM_TEAMS_TENANT_MANAGE = "teams:tenant:manage"
PERM_TEAMS_ADMIN = "teams:admin"

# Legacy permission for backward compatibility
CONNECTOR_AUTHORIZE = "connectors.authorize"

# Environment configuration
TEAMS_CLIENT_ID = os.environ.get("TEAMS_CLIENT_ID")
TEAMS_CLIENT_SECRET = os.environ.get("TEAMS_CLIENT_SECRET")
TEAMS_REDIRECT_URI = os.environ.get("TEAMS_REDIRECT_URI")

# Log at debug level for unconfigured optional integrations
if not TEAMS_CLIENT_ID:
    logger.debug("TEAMS_CLIENT_ID not configured - Teams OAuth disabled")
if not TEAMS_CLIENT_SECRET:
    logger.debug("TEAMS_CLIENT_SECRET not configured - Teams OAuth disabled")

# Default OAuth scopes for Aragora Teams app
DEFAULT_SCOPES = "https://graph.microsoft.com/.default offline_access"
TEAMS_SCOPES = os.environ.get("TEAMS_SCOPES", DEFAULT_SCOPES)

# Microsoft OAuth URLs
MS_OAUTH_AUTHORIZE_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
MS_OAUTH_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"


def _get_state_store():
    """Get the centralized OAuth state store."""
    from aragora.server.oauth_state_store import get_oauth_state_store

    return get_oauth_state_store()


class TeamsOAuthHandler(SecureHandler):
    """Handler for Microsoft Teams OAuth installation flow.

    RBAC Protection:
    - /install: Requires teams:oauth:install OR connectors.authorize permission
    - /callback: No auth (OAuth callback from Microsoft, state validated)
    - /refresh: Requires teams:tenant:manage permission
    - /disconnect: Requires teams:oauth:disconnect permission
    - /tenants: Requires teams:tenant:manage permission (list/manage tenants)

    Security Notes:
    - OAuth callback validates state token to prevent CSRF
    - All authenticated endpoints require valid JWT
    - Tenant operations require specific tenant management permissions
    - Admin operations require teams:admin permission
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    RESOURCE_TYPE = "connector"

    ROUTES = [
        "/api/integrations/teams/install",
        "/api/integrations/teams/callback",
        "/api/integrations/teams/refresh",
        "/api/integrations/teams/disconnect",
        "/api/integrations/teams/tenants",
    ]

    # Route patterns for dynamic paths
    ROUTE_PATTERNS = [
        r"/api/integrations/teams/tenants/([^/]+)",
        r"/api/integrations/teams/tenants/([^/]+)/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        import re

        if path in self.ROUTES:
            return True
        # Check dynamic patterns
        for pattern in self.ROUTE_PATTERNS:
            if re.match(pattern, path):
                return True
        return False

    def _check_permission(
        self,
        auth_context: Any,
        permission: str,
        fallback_permission: str | None = None,
    ) -> bool:
        """Check if user has required permission with optional fallback.

        Args:
            auth_context: User's authorization context
            permission: Primary permission to check
            fallback_permission: Optional fallback permission (for backward compatibility)

        Returns:
            True if permission granted

        Raises:
            ForbiddenError: If permission denied
        """
        try:
            self.check_permission(auth_context, permission)
            return True
        except (ForbiddenError, PermissionError):
            if fallback_permission:
                try:
                    self.check_permission(auth_context, fallback_permission)
                    return True
                except (ForbiddenError, PermissionError):
                    pass
            raise ForbiddenError("Permission denied")

    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """BaseHandler-compatible entry point.

        This wrapper delegates to dispatch() which supports the OAuth-style
        calling convention used by this handler's tests and callers.

        Args:
            path: Request path (e.g., "/api/integrations/teams/install")
            query_params: Query parameters dict
            handler: HTTP request handler

        Returns:
            HandlerResult or None if not handled
        """
        return await self.dispatch(
            method="GET",  # BaseHandler only calls handle() for GET
            path=path,
            query_params=query_params,
            handler=handler,
        )

    async def dispatch(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        query_params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        handler: Any | None = None,
    ) -> HandlerResult:
        """Route OAuth requests to appropriate methods.

        This is the primary entry point supporting the OAuth-style calling convention:
            dispatch(method, path, body=None, query_params=None, headers=None, handler=None)

        For BaseHandler compatibility, use handle(path, query_params, handler) which
        delegates to this method.

        RBAC enforcement:
        - /install: Requires teams:oauth:install (or legacy connectors.authorize)
        - /callback: No auth (OAuth redirect from Microsoft, state validated)
        - /refresh: Requires teams:tenant:manage permission
        - /disconnect: Requires teams:oauth:disconnect permission
        - /tenants: Requires teams:tenant:manage permission
        - /tenants/{id}: Requires teams:tenant:manage permission
        - /tenants/{id}/status: Requires teams:tenant:manage permission
        """
        import re

        query_params_dict = query_params or {}
        body = body or {}

        # OAuth callback from Microsoft - no auth required (external redirect)
        # Security: State token is validated in _handle_callback to prevent CSRF
        if path == "/api/integrations/teams/callback":
            if method == "GET":
                return await self._handle_callback(query_params_dict)
            return error_response("Method not allowed", 405)

        # All other routes require authentication
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except (UnauthorizedError, PermissionError, ValueError, RuntimeError) as e:
            logger.debug(f"Teams OAuth auth failed: {e}")
            return error_response("Authentication required", 401)

        # Install endpoint - initiate OAuth flow
        if path == "/api/integrations/teams/install":
            if method == "GET":
                try:
                    # Check teams:oauth:install with fallback to legacy connector permission
                    self._check_permission(
                        auth_context,
                        PERM_TEAMS_OAUTH_INSTALL,
                        fallback_permission=CONNECTOR_AUTHORIZE,
                    )
                except ForbiddenError:
                    return error_response(
                        "Permission denied", 403
                    )
                return await self._handle_install(query_params_dict)
            return error_response("Method not allowed", 405)

        # Refresh endpoint - refresh tokens for a tenant
        if path == "/api/integrations/teams/refresh":
            if method == "POST":
                try:
                    self._check_permission(auth_context, PERM_TEAMS_TENANT_MANAGE)
                except ForbiddenError:
                    return error_response(
                        "Permission denied", 403
                    )
                return await self._handle_refresh(body)
            return error_response("Method not allowed", 405)

        # Disconnect endpoint - disconnect a tenant
        if path == "/api/integrations/teams/disconnect":
            if method == "POST":
                try:
                    self._check_permission(auth_context, PERM_TEAMS_OAUTH_DISCONNECT)
                except ForbiddenError:
                    return error_response(
                        "Permission denied", 403
                    )
                return await self._handle_disconnect(body)
            return error_response("Method not allowed", 405)

        # Tenants list endpoint
        if path == "/api/integrations/teams/tenants":
            if method == "GET":
                try:
                    self._check_permission(auth_context, PERM_TEAMS_TENANT_MANAGE)
                except ForbiddenError:
                    return error_response(
                        "Permission denied", 403
                    )
                return await self._handle_list_tenants()
            return error_response("Method not allowed", 405)

        # Check for dynamic tenant routes
        # GET/DELETE /api/integrations/teams/tenants/{tenant_id}
        tenant_match = re.match(r"/api/integrations/teams/tenants/([^/]+)$", path)
        if tenant_match:
            tenant_id = tenant_match.group(1)
            if method == "GET":
                try:
                    self._check_permission(auth_context, PERM_TEAMS_TENANT_MANAGE)
                except ForbiddenError:
                    return error_response(
                        "Permission denied", 403
                    )
                return await self._handle_get_tenant(tenant_id)
            if method == "DELETE":
                try:
                    self._check_permission(auth_context, PERM_TEAMS_OAUTH_DISCONNECT)
                except ForbiddenError:
                    return error_response(
                        "Permission denied", 403
                    )
                return await self._handle_disconnect({"tenant_id": tenant_id})
            return error_response("Method not allowed", 405)

        # GET /api/integrations/teams/tenants/{tenant_id}/status
        status_match = re.match(r"/api/integrations/teams/tenants/([^/]+)/status$", path)
        if status_match:
            tenant_id = status_match.group(1)
            if method == "GET":
                try:
                    self._check_permission(auth_context, PERM_TEAMS_TENANT_MANAGE)
                except ForbiddenError:
                    return error_response(
                        "Permission denied", 403
                    )
                return await self._handle_tenant_status(tenant_id)
            return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    async def _handle_install(self, query_params: dict[str, str]) -> HandlerResult:
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

        org_id = query_params.get("org_id")

        # Generate state using centralized OAuth state store
        state_store = _get_state_store()
        try:
            state = state_store.generate(metadata={"org_id": org_id, "provider": "teams"})
        except (RuntimeError, ValueError, OSError, TypeError) as e:
            logger.error(f"Failed to generate OAuth state: {e}")
            return error_response("Failed to initialize OAuth flow", 503)

        # Build OAuth URL
        redirect_uri = TEAMS_REDIRECT_URI
        if not redirect_uri:
            # Development fallback only - restrict to localhost to prevent open redirect
            host = query_params.get("host", "localhost:8080")
            if not host.startswith(("localhost", "127.0.0.1", "[::1]")):
                return error_response("Only localhost allowed without TEAMS_REDIRECT_URI", 400)
            scheme = "http"
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

    async def _handle_callback(self, query_params: dict[str, str]) -> HandlerResult:
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

        # Verify state token using centralized state store
        state_store = _get_state_store()
        state_data = state_store.validate_and_consume(state)
        if not state_data:
            return error_response("Invalid or expired state token", 400)

        org_id = state_data.metadata.get("org_id") if state_data.metadata else None

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
            logger.error(f"Teams token exchange failed: status={e.response.status_code}")
            return error_response("Token exchange failed", 500)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.error(f"Teams token exchange failed: {e}")
            return error_response("Token exchange failed", 500)

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

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
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

    async def _handle_refresh(self, body: dict[str, Any]) -> HandlerResult:
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
            logger.error(f"Teams token refresh failed: status={e.response.status_code}")
            return error_response("Token refresh failed", 500)
        except (ConnectionError, TimeoutError, OSError, ValueError) as e:
            logger.error(f"Teams token refresh failed: {e}")
            return error_response("Token refresh failed", 500)

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

    async def _handle_disconnect(self, body: dict[str, Any]) -> HandlerResult:
        """
        Disconnect a Teams tenant.

        Request body:
            tenant_id: Azure AD tenant ID to disconnect

        RBAC: Requires teams:oauth:disconnect permission
        """
        tenant_id = body.get("tenant_id")
        if not tenant_id:
            return error_response("Missing tenant_id", 400)

        try:
            from aragora.storage.teams_tenant_store import get_teams_tenant_store

            store = get_teams_tenant_store()
            tenant = store.get(tenant_id)

            if not tenant:
                return error_response("Tenant not found", 404)

            # Deactivate the tenant
            tenant.is_active = False
            if not store.save(tenant):
                return error_response("Failed to disconnect tenant", 500)

            logger.info(f"Teams tenant disconnected: {tenant.tenant_name} ({tenant_id})")

            return json_response(
                {
                    "success": True,
                    "tenant_id": tenant_id,
                    "tenant_name": tenant.tenant_name,
                    "message": "Tenant disconnected successfully",
                }
            )

        except ImportError as e:
            logger.error(f"Tenant store not available: {e}")
            return error_response("Tenant storage not available", 503)
        except (KeyError, ValueError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to disconnect tenant: {e}")
            return error_response("Failed to disconnect tenant", 500)

    async def _handle_list_tenants(self) -> HandlerResult:
        """
        List all Teams tenants with their status.

        RBAC: Requires teams:tenant:manage permission

        Returns:
            List of tenants with id, name, is_active, token status
        """
        try:
            from aragora.storage.teams_tenant_store import get_teams_tenant_store

            store = get_teams_tenant_store()
            tenants = store.list_all()

            tenant_list = []
            current_time = time.time()

            for t in tenants:
                # Determine token health
                token_status = "valid"
                if t.expires_at:
                    if t.expires_at < current_time:
                        token_status = "expired"
                    elif t.expires_at < current_time + 3600:
                        token_status = "expiring_soon"

                tenant_list.append(
                    {
                        "tenant_id": t.tenant_id,
                        "tenant_name": t.tenant_name,
                        "is_active": t.is_active,
                        "token_status": token_status,
                        "expires_at": t.expires_at,
                        "installed_at": t.installed_at,
                        "installed_by": t.installed_by,
                        "scopes": t.scopes,
                        "aragora_org_id": t.aragora_org_id,
                    }
                )

            logger.info(f"Listed {len(tenant_list)} Teams tenants")
            return json_response(
                {
                    "tenants": tenant_list,
                    "total": len(tenant_list),
                }
            )

        except ImportError as e:
            logger.error(f"Tenant store not available: {e}")
            return error_response("Tenant storage not available", 503)
        except (KeyError, ValueError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to list tenants: {e}")
            return error_response("Failed to list tenants", 500)

    async def _handle_get_tenant(self, tenant_id: str) -> HandlerResult:
        """
        Get details for a specific Teams tenant.

        Args:
            tenant_id: The Azure AD tenant ID

        RBAC: Requires teams:tenant:manage permission

        Returns:
            Tenant details including token status
        """
        try:
            from aragora.storage.teams_tenant_store import get_teams_tenant_store

            store = get_teams_tenant_store()
            tenant = store.get(tenant_id)

            if not tenant:
                return error_response(f"Tenant {tenant_id} not found", 404)

            current_time = time.time()

            # Determine token health
            token_status = "valid"
            expires_in_seconds = None
            if tenant.expires_at:
                expires_in_seconds = int(tenant.expires_at - current_time)
                if expires_in_seconds < 0:
                    token_status = "expired"
                elif expires_in_seconds < 3600:
                    token_status = "expiring_soon"
                elif expires_in_seconds < 86400:
                    token_status = "expiring_today"

            # Check if refresh token is available
            has_refresh_token = bool(tenant.refresh_token)

            tenant_data = {
                "tenant_id": tenant.tenant_id,
                "tenant_name": tenant.tenant_name,
                "is_active": tenant.is_active,
                "token_status": token_status,
                "expires_at": tenant.expires_at,
                "expires_in_seconds": expires_in_seconds,
                "has_refresh_token": has_refresh_token,
                "scopes": tenant.scopes,
                "installed_at": tenant.installed_at,
                "installed_by": tenant.installed_by,
                "bot_id": tenant.bot_id,
                "aragora_org_id": tenant.aragora_org_id,
            }

            logger.debug(f"Retrieved tenant {tenant_id}: {token_status}")
            return json_response(tenant_data)

        except ImportError as e:
            logger.error(f"Tenant store not available: {e}")
            return error_response("Tenant storage not available", 503)
        except (KeyError, ValueError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to get tenant: {e}")
            return error_response("Failed to get tenant", 500)

    async def _handle_tenant_status(self, tenant_id: str) -> HandlerResult:
        """
        Get detailed token status for a specific tenant.

        Args:
            tenant_id: The Azure AD tenant ID

        RBAC: Requires teams:tenant:manage permission

        Returns:
            Token health details including validity, expiration, scopes
        """
        try:
            from aragora.storage.teams_tenant_store import get_teams_tenant_store

            store = get_teams_tenant_store()
            tenant = store.get(tenant_id)

            if not tenant:
                return error_response(f"Tenant {tenant_id} not found", 404)

            current_time = time.time()

            # Determine token health
            token_status = "valid"
            expires_in_seconds = None
            if tenant.expires_at:
                expires_in_seconds = int(tenant.expires_at - current_time)
                if expires_in_seconds < 0:
                    token_status = "expired"
                elif expires_in_seconds < 3600:
                    token_status = "expiring_soon"
                elif expires_in_seconds < 86400:
                    token_status = "expiring_today"

            # Check if refresh token is available
            has_refresh_token = bool(tenant.refresh_token)

            # Calculate time until refresh recommended
            refresh_recommended_in = None
            if expires_in_seconds and expires_in_seconds > 0:
                # Recommend refresh when 50% of token lifetime has passed
                refresh_recommended_in = max(0, expires_in_seconds - 1800)

            status_data = {
                "tenant_id": tenant.tenant_id,
                "tenant_name": tenant.tenant_name,
                "is_active": tenant.is_active,
                "token_status": token_status,
                "expires_at": tenant.expires_at,
                "expires_in_seconds": expires_in_seconds,
                "has_refresh_token": has_refresh_token,
                "refresh_recommended_in": refresh_recommended_in,
                "can_refresh": has_refresh_token and tenant.is_active,
                "scopes": tenant.scopes,
            }

            logger.debug(f"Token status for tenant {tenant_id}: {token_status}")
            return json_response(status_data)

        except ImportError as e:
            logger.error(f"Tenant store not available: {e}")
            return error_response("Tenant storage not available", 503)
        except (KeyError, ValueError, OSError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to get tenant status: {e}")
            return error_response("Failed to get tenant status", 500)


# Handler factory function for registration
def create_teams_oauth_handler(server_context: Any) -> TeamsOAuthHandler:
    """Factory function for handler registration."""
    return TeamsOAuthHandler(server_context)
