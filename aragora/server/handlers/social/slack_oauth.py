"""
Slack OAuth Handler for app installation flow.

Endpoints:
- GET  /api/integrations/slack/install   - Redirect to Slack OAuth
- GET  /api/integrations/slack/callback  - Handle OAuth callback
- POST /api/integrations/slack/uninstall - Handle app removal

Environment Variables:
- SLACK_CLIENT_ID: App client ID
- SLACK_CLIENT_SECRET: App client secret
- SLACK_REDIRECT_URI: OAuth callback URL (REQUIRED in production, falls back to localhost in dev)
- SLACK_SCOPES: OAuth scopes (default: channels:history,chat:write,commands,users:read)
- ARAGORA_ENV: Environment mode ('production' enforces SLACK_REDIRECT_URI)

See: https://api.slack.com/authentication/oauth-v2
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

from aragora.server.oauth_state_store import OAUTH_STATE_TTL_SECONDS

from ..base import (
    HandlerResult,
    error_response,
    json_response,
)
from ..secure import SecureHandler, UnauthorizedError

# RBAC Permissions for Slack OAuth operations
CONNECTOR_READ = "connectors.read"
CONNECTOR_AUTHORIZE = "connectors.authorize"

# Environment configuration
SLACK_CLIENT_ID = os.environ.get("SLACK_CLIENT_ID", "")
SLACK_CLIENT_SECRET = os.environ.get("SLACK_CLIENT_SECRET", "")
SLACK_REDIRECT_URI = os.environ.get("SLACK_REDIRECT_URI", "")
ARAGORA_ENV = os.environ.get("ARAGORA_ENV", "development")

# Default OAuth scopes for Aragora Slack app
DEFAULT_SCOPES = "channels:history,chat:write,commands,users:read,team:read,channels:read"
SLACK_SCOPES = os.environ.get("SLACK_SCOPES", DEFAULT_SCOPES)

# Slack OAuth URLs
SLACK_OAUTH_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
SLACK_OAUTH_TOKEN_URL = "https://slack.com/api/oauth.v2.access"

# Scope descriptions for consent preview page
SCOPE_DESCRIPTIONS = {
    "channels:history": {
        "name": "Read Channel Messages",
        "description": "Access message history to provide context for debates and discussions",
        "required": True,
        "icon": "📖",
    },
    "chat:write": {
        "name": "Send Messages",
        "description": "Post debate results, summaries, and AI-generated insights to channels",
        "required": True,
        "icon": "✍️",
    },
    "commands": {
        "name": "Slash Commands",
        "description": "Respond to /aragora commands for quick access to debates",
        "required": False,
        "icon": "⚡",
    },
    "users:read": {
        "name": "View User Information",
        "description": "Identify participants in discussions by name and profile",
        "required": True,
        "icon": "👥",
    },
    "team:read": {
        "name": "View Workspace Info",
        "description": "Access workspace metadata for configuration and analytics",
        "required": False,
        "icon": "🏢",
    },
    "channels:read": {
        "name": "List Channels",
        "description": "View available channels to select where Aragora can operate",
        "required": True,
        "icon": "📋",
    },
}

# Lazy import for audit logger
_slack_oauth_audit: Any = None

# Legacy in-memory fallback for tests/compatibility
_oauth_states_fallback: dict[str, dict[str, Any]] = {}


def _cleanup_oauth_states_fallback(now: Optional[float] = None) -> None:
    """Remove expired fallback OAuth states."""
    now = now or time.time()
    expired = [
        state
        for state, data in _oauth_states_fallback.items()
        if now - data.get("created_at", now) > OAUTH_STATE_TTL_SECONDS
    ]
    for state in expired:
        _oauth_states_fallback.pop(state, None)


def _get_oauth_audit_logger() -> Any:
    """Get or create Slack audit logger for OAuth (lazy initialization)."""
    global _slack_oauth_audit
    if _slack_oauth_audit is None:
        try:
            from aragora.audit.slack_audit import get_slack_audit_logger

            _slack_oauth_audit = get_slack_audit_logger()
        except Exception as e:
            logger.debug(f"Slack OAuth audit logger not available: {e}")
            _slack_oauth_audit = None
    return _slack_oauth_audit


def _get_state_store():
    """Get the centralized OAuth state store."""
    from aragora.server.oauth_state_store import get_oauth_state_store

    return get_oauth_state_store()


class SlackOAuthHandler(SecureHandler):
    """Handler for Slack OAuth installation flow.

    RBAC Protection:
    - /install: Requires connector:authorize permission
    - /preview: Requires connector:read permission
    - /callback: No auth (OAuth callback from Slack)
    - /uninstall: Verified via Slack signature (webhook)
    """

    RESOURCE_TYPE = "connector"

    ROUTES = [
        "/api/integrations/slack/install",
        "/api/integrations/slack/preview",
        "/api/integrations/slack/callback",
        "/api/integrations/slack/uninstall",
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
        handler: Optional[Any] = None,
    ) -> HandlerResult:
        """Route OAuth requests to appropriate methods.

        RBAC enforcement:
        - /install and /preview require authentication
        - /callback is unauthenticated (OAuth redirect from Slack)
        - /uninstall uses Slack signature verification (webhook)
        """
        query_params = query_params or {}
        headers = headers or {}
        body = body or {}

        # OAuth callback from Slack - no auth required (external redirect)
        if path == "/api/integrations/slack/callback":
            if method == "GET":
                return await self._handle_callback(query_params)
            return error_response("Method not allowed", 405)

        # Uninstall webhook from Slack - verified via Slack signature
        if path == "/api/integrations/slack/uninstall":
            if method == "POST":
                return await self._handle_uninstall(body, headers or {})
            return error_response("Method not allowed", 405)

        # All other routes require authentication
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
        except (UnauthorizedError, Exception) as e:
            logger.debug(f"Slack OAuth auth failed: {e}")
            return error_response("Authentication required", 401)

        if path == "/api/integrations/slack/install":
            if method == "GET":
                # Require connector:authorize permission
                try:
                    self.check_permission(auth_context, CONNECTOR_AUTHORIZE)
                except Exception as e:
                    logger.warning(f"Permission denied for Slack install: {e}")
                    return error_response("Permission denied: connector:authorize required", 403)
                return await self._handle_install(query_params)
            return error_response("Method not allowed", 405)

        elif path == "/api/integrations/slack/preview":
            if method == "GET":
                # Require connector:read permission
                try:
                    self.check_permission(auth_context, CONNECTOR_READ)
                except Exception as e:
                    logger.warning(f"Permission denied for Slack preview: {e}")
                    return error_response("Permission denied: connector:read required", 403)
                return await self._handle_preview(query_params)
            return error_response("Method not allowed", 405)

        return error_response("Not found", 404)

    async def _handle_install(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Initiate Slack OAuth installation flow.

        Redirects user to Slack's OAuth consent page.
        Optional query params:
            tenant_id: Aragora tenant to link workspace to
        """
        if not SLACK_CLIENT_ID:
            return error_response(
                "Slack OAuth not configured. Set SLACK_CLIENT_ID environment variable.",
                503,
            )

        tenant_id = query_params.get("tenant_id")

        # Generate state using centralized OAuth state store
        state_store = _get_state_store()
        try:
            state = state_store.generate(metadata={"tenant_id": tenant_id, "provider": "slack"})
        except Exception as e:
            logger.error(f"Failed to generate OAuth state: {e}")
            return error_response("Failed to initialize OAuth flow", 503)

        # Build OAuth URL
        redirect_uri = SLACK_REDIRECT_URI
        if not redirect_uri:
            # SLACK_REDIRECT_URI is required in production to prevent open redirect attacks
            if ARAGORA_ENV == "production":
                return error_response(
                    "SLACK_REDIRECT_URI must be configured in production",
                    500,
                )
            # Development fallback only - construct from request host parameter
            host = query_params.get("host", "localhost:8080")
            scheme = "https" if "localhost" not in host else "http"
            redirect_uri = f"{scheme}://{host}/api/integrations/slack/callback"
            logger.warning(f"Using fallback redirect_uri in development: {redirect_uri}")

        oauth_params = {
            "client_id": SLACK_CLIENT_ID,
            "scope": SLACK_SCOPES,
            "redirect_uri": redirect_uri,
            "state": state,
        }

        oauth_url = f"{SLACK_OAUTH_AUTHORIZE_URL}?{urlencode(oauth_params)}"

        _cleanup_oauth_states_fallback()
        _oauth_states_fallback[state] = {
            "created_at": time.time(),
            "tenant_id": tenant_id,
            "provider": "slack",
        }

        logger.info(f"Initiating Slack OAuth flow (state: {state[:8]}...)")

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

    async def _handle_preview(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Display consent preview page before Slack OAuth.

        Shows users what permissions Aragora needs and why, before
        redirecting to Slack's authorization page.

        Query params:
            tenant_id: Optional tenant to link workspace to
        """
        if not SLACK_CLIENT_ID:
            return error_response(
                "Slack OAuth not configured. Set SLACK_CLIENT_ID environment variable.",
                503,
            )

        tenant_id = query_params.get("tenant_id", "")

        # Build scope information for display
        current_scopes = SLACK_SCOPES.split(",")
        required_scopes = []
        optional_scopes = []

        for scope in current_scopes:
            scope = scope.strip()
            if scope in SCOPE_DESCRIPTIONS:
                desc = SCOPE_DESCRIPTIONS[scope]
                scope_info = {
                    "scope": scope,
                    "name": desc["name"],
                    "description": desc["description"],
                    "icon": desc.get("icon", ""),
                }
                if desc.get("required", True):
                    required_scopes.append(scope_info)
                else:
                    optional_scopes.append(scope_info)
            else:
                # Unknown scope - show as required
                required_scopes.append(
                    {
                        "scope": scope,
                        "name": scope.replace(":", " ").replace("_", " ").title(),
                        "description": f"Permission: {scope}",
                        "icon": "",
                    }
                )

        # Build install URL with tenant_id
        install_url = "/api/integrations/slack/install"
        if tenant_id:
            install_url += f"?tenant_id={tenant_id}"

        # Generate HTML consent preview page
        html = self._render_consent_preview(
            required_scopes=required_scopes,
            optional_scopes=optional_scopes,
            install_url=install_url,
        )

        return HandlerResult(
            status_code=200,
            content_type="text/html; charset=utf-8",
            body=html.encode("utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    def _render_consent_preview(
        self,
        required_scopes: list,
        optional_scopes: list,
        install_url: str,
    ) -> str:
        """Render the consent preview HTML page."""
        required_html = ""
        for s in required_scopes:
            icon = s["icon"] if s["icon"] else "&#128274;"
            required_html += f"""
            <div class="scope-item">
                <span class="scope-icon">{icon}</span>
                <div class="scope-details">
                    <div class="scope-name">{s["name"]}</div>
                    <div class="scope-desc">{s["description"]}</div>
                </div>
                <span class="scope-badge required">Required</span>
            </div>
            """

        optional_html = ""
        for s in optional_scopes:
            icon = s["icon"] if s["icon"] else "&#128274;"
            optional_html += f"""
            <div class="scope-item">
                <span class="scope-icon">{icon}</span>
                <div class="scope-details">
                    <div class="scope-name">{s["name"]}</div>
                    <div class="scope-desc">{s["description"]}</div>
                </div>
                <span class="scope-badge optional">Optional</span>
            </div>
            """

        optional_section = ""
        if optional_html:
            optional_section = (
                "<div class='section-title' style='margin-top: 24px;'>Optional Features</div>"
                + optional_html
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Install Aragora in Slack</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            overflow: hidden;
        }}
        .header {{
            background: #4A154B;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header p {{ opacity: 0.9; font-size: 14px; }}
        .logo {{
            width: 60px; height: 60px;
            background: white;
            border-radius: 12px;
            margin: 0 auto 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
        }}
        .content {{ padding: 30px; }}
        .section-title {{
            font-size: 14px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
        }}
        .scope-item {{
            display: flex;
            align-items: flex-start;
            gap: 12px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 10px;
        }}
        .scope-icon {{ font-size: 20px; flex-shrink: 0; }}
        .scope-details {{ flex: 1; }}
        .scope-name {{ font-weight: 600; color: #1a1a1a; margin-bottom: 4px; }}
        .scope-desc {{ font-size: 13px; color: #666; line-height: 1.4; }}
        .scope-badge {{
            font-size: 11px;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            flex-shrink: 0;
        }}
        .scope-badge.required {{ background: #e3f2fd; color: #1565c0; }}
        .scope-badge.optional {{ background: #f3e5f5; color: #7b1fa2; }}
        .data-notice {{
            background: #fff8e1;
            border: 1px solid #ffcc02;
            border-radius: 8px;
            padding: 16px;
            margin: 20px 0;
        }}
        .data-notice h4 {{ color: #f57c00; font-size: 14px; margin-bottom: 8px; }}
        .data-notice ul {{ font-size: 13px; color: #666; margin-left: 20px; }}
        .data-notice li {{ margin-bottom: 4px; }}
        .actions {{ padding: 20px 30px 30px; border-top: 1px solid #eee; }}
        .btn {{
            display: block;
            width: 100%;
            padding: 14px 24px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .btn-primary {{ background: #4A154B; color: white; }}
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74, 21, 75, 0.3);
        }}
        .btn-secondary {{ background: transparent; color: #666; margin-top: 12px; }}
        .btn-secondary:hover {{ color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">&#129302;</div>
            <h1>Install Aragora</h1>
            <p>AI-powered multi-agent debates for your Slack workspace</p>
        </div>
        <div class="content">
            <div class="section-title">Required Permissions</div>
            {required_html}
            {optional_section}
            <div class="data-notice">
                <h4>&#128274; How We Handle Your Data</h4>
                <ul>
                    <li>Messages are analyzed only for the current debate session</li>
                    <li>User data is used solely for participant identification</li>
                    <li>No message history is stored after debate completion</li>
                    <li>All data is encrypted in transit and at rest</li>
                </ul>
            </div>
        </div>
        <div class="actions">
            <a href="{install_url}" class="btn btn-primary">
                &#128241; Continue to Slack Authorization
            </a>
            <a href="javascript:history.back()" class="btn btn-secondary">Cancel</a>
        </div>
    </div>
</body>
</html>"""

    async def _handle_callback(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Handle OAuth callback from Slack.

        Query params:
            code: Authorization code from Slack
            state: State token for CSRF verification
            error: Error code if user denied
        """
        # Check for error from Slack
        if "error" in query_params:
            error_code = query_params.get("error")
            logger.warning(f"Slack OAuth error: {error_code}")
            # Audit log OAuth denial
            audit = _get_oauth_audit_logger()
            if audit:
                audit.log_oauth(
                    workspace_id="",
                    action="install",
                    success=False,
                    error=f"User denied: {error_code}",
                )
            return error_response(f"Slack authorization denied: {error_code}", 400)

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
            _cleanup_oauth_states_fallback()
            state_data = _oauth_states_fallback.pop(state, None)
        if not state_data:
            return error_response("Invalid or expired state token", 400)

        if isinstance(state_data, dict):
            tenant_id = state_data.get("tenant_id")
        else:
            metadata = getattr(state_data, "metadata", None)
            tenant_id = metadata.get("tenant_id") if isinstance(metadata, dict) else None

        if not SLACK_CLIENT_ID or not SLACK_CLIENT_SECRET:
            return error_response("Slack OAuth not configured", 503)

        # Exchange code for access token with retry logic
        request_id = secrets.token_hex(8)
        max_retries = 3
        retry_delay = 1.0  # seconds

        try:
            import asyncio

            import httpx

            data = None
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.post(
                            SLACK_OAUTH_TOKEN_URL,
                            data={
                                "client_id": SLACK_CLIENT_ID,
                                "client_secret": SLACK_CLIENT_SECRET,
                                "code": code,
                                "redirect_uri": SLACK_REDIRECT_URI,
                            },
                        )

                        # Check for retryable status codes (safely handle mocked responses)
                        status_code = getattr(response, "status_code", 200)
                        if isinstance(status_code, int):
                            if status_code == 429:
                                # Rate limited - wait and retry
                                retry_after = int(
                                    response.headers.get("Retry-After", retry_delay * 2)
                                )
                                logger.warning(
                                    f"[{request_id}] Slack OAuth rate limited, "
                                    f"retrying in {retry_after}s (attempt {attempt + 1}/{max_retries})"
                                )
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_after)
                                    continue

                            if status_code >= 500:
                                # Server error - retry with backoff
                                logger.warning(
                                    f"[{request_id}] Slack OAuth server error {status_code}, "
                                    f"retrying (attempt {attempt + 1}/{max_retries})"
                                )
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_delay * (2**attempt))
                                    continue

                        response.raise_for_status()
                        data = response.json()
                        break  # Success

                except httpx.TimeoutException as e:
                    last_error = e
                    logger.warning(
                        f"[{request_id}] Slack OAuth timeout, "
                        f"retrying (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue

                except httpx.ConnectError as e:
                    last_error = e
                    logger.warning(
                        f"[{request_id}] Slack OAuth connection error, "
                        f"retrying (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2**attempt))
                        continue

            if data is None:
                logger.error(
                    f"[{request_id}] Slack token exchange failed after {max_retries} attempts: {last_error}"
                )
                return error_response(f"Token exchange failed after retries: {last_error}", 500)

        except ImportError:
            return error_response("httpx not available", 503)
        except Exception as e:
            logger.error(f"[{request_id}] Slack token exchange failed: {e}")
            return error_response(f"Token exchange failed: {e}", 500)

        if not data.get("ok"):
            error_msg = data.get("error", "Unknown error")
            logger.error(f"Slack OAuth failed: {error_msg}")
            return error_response(f"Slack OAuth failed: {error_msg}", 400)

        # Extract workspace info
        access_token = data.get("access_token")
        team = data.get("team", {})
        bot_user_id = data.get("bot_user_id", "")
        authed_user = data.get("authed_user", {})
        scope = data.get("scope", "")

        # Extract token refresh data (if available)
        refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in")  # Seconds until expiration
        token_expires_at = None
        if expires_in:
            token_expires_at = time.time() + expires_in

        workspace_id = team.get("id", "")
        workspace_name = team.get("name", "Unknown")
        installed_by = authed_user.get("id")

        if not workspace_id or not access_token:
            return error_response("Invalid response from Slack", 500)

        # Store workspace credentials
        try:
            from aragora.storage.slack_workspace_store import (
                SlackWorkspace,
                get_slack_workspace_store,
            )

            workspace = SlackWorkspace(
                workspace_id=workspace_id,
                workspace_name=workspace_name,
                access_token=access_token,
                bot_user_id=bot_user_id,
                installed_at=time.time(),
                installed_by=installed_by,
                scopes=scope.split(",") if scope else [],
                tenant_id=tenant_id,
                is_active=True,
                refresh_token=refresh_token,
                token_expires_at=token_expires_at,
            )

            store = get_slack_workspace_store()
            if not store.save(workspace):
                # Audit log save failure
                audit = _get_oauth_audit_logger()
                if audit:
                    audit.log_oauth(
                        workspace_id=workspace_id,
                        action="install",
                        success=False,
                        error="Failed to save workspace credentials",
                        user_id=installed_by or "",
                    )
                return error_response("Failed to save workspace", 500)

            logger.info(f"Slack workspace installed: {workspace_name} ({workspace_id})")

            # Audit log successful installation
            audit = _get_oauth_audit_logger()
            if audit:
                audit.log_oauth(
                    workspace_id=workspace_id,
                    action="install",
                    success=True,
                    user_id=installed_by or "",
                    scopes=scope.split(",") if scope else [],
                )

        except ImportError as e:
            logger.error(f"Workspace store not available: {e}")
            # Audit log storage unavailable error
            audit = _get_oauth_audit_logger()
            if audit:
                audit.log_oauth(
                    workspace_id=workspace_id,
                    action="install",
                    success=False,
                    error=f"Workspace storage not available: {e}",
                )
            return error_response("Workspace storage not available", 503)

        # Return success page
        success_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Aragora - Slack Connected</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
                .workspace {{
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
                <p class="workspace">{workspace_name}</p>
                <p>You can close this window and return to Slack.</p>
            </div>
        </body>
        </html>
        """

        return HandlerResult(
            status_code=200,
            content_type="text/html",
            body=success_html.encode("utf-8"),
        )

    async def _handle_uninstall(
        self, body: Dict[str, Any], headers: Dict[str, str]
    ) -> HandlerResult:
        """
        Handle app uninstallation webhook from Slack.

        This is called by Slack when a user uninstalls the app.
        Verifies the request signature using the Slack signing secret.
        """
        # Verify signature in production
        signing_secret = os.environ.get("SLACK_SIGNING_SECRET", "")
        if signing_secret:
            timestamp = headers.get("x-slack-request-timestamp", "")
            signature = headers.get("x-slack-signature", "")

            if not timestamp or not signature:
                logger.warning("Missing Slack signature headers")
                return error_response("Missing signature", 401)

            # Check timestamp is recent (within 5 minutes)
            try:
                request_time = int(timestamp)
                if abs(time.time() - request_time) > 300:
                    logger.warning("Slack request timestamp too old")
                    return error_response("Request expired", 401)
            except ValueError:
                return error_response("Invalid timestamp", 401)

            # Verify signature
            import hmac
            import hashlib

            sig_basestring = f"v0:{timestamp}:{json.dumps(body, separators=(',', ':'))}"
            computed_sig = (
                "v0="
                + hmac.new(
                    signing_secret.encode(),
                    sig_basestring.encode(),
                    hashlib.sha256,
                ).hexdigest()
            )

            if not hmac.compare_digest(signature, computed_sig):
                logger.warning("Invalid Slack signature")
                return error_response("Invalid signature", 401)

        event = body.get("event", {})
        event_type = event.get("type")

        if event_type == "app_uninstalled":
            workspace_id = body.get("team_id") or event.get("team_id")

            if workspace_id:
                try:
                    from aragora.storage.slack_workspace_store import (
                        get_slack_workspace_store,
                    )

                    store = get_slack_workspace_store()
                    store.deactivate(workspace_id)
                    logger.info(f"Slack workspace uninstalled: {workspace_id}")

                    # Audit log uninstallation
                    audit = _get_oauth_audit_logger()
                    if audit:
                        audit.log_oauth(
                            workspace_id=workspace_id,
                            action="uninstall",
                            success=True,
                        )

                except ImportError:
                    logger.warning("Could not deactivate workspace - store unavailable")

        elif event_type == "tokens_revoked":
            workspace_id = body.get("team_id")
            tokens = event.get("tokens", {})
            bot_tokens = tokens.get("bot", [])

            if workspace_id and bot_tokens:
                try:
                    from aragora.storage.slack_workspace_store import (
                        get_slack_workspace_store,
                    )

                    store = get_slack_workspace_store()
                    store.deactivate(workspace_id)
                    logger.info(f"Slack tokens revoked for workspace: {workspace_id}")

                    # Audit log token revocation
                    audit = _get_oauth_audit_logger()
                    if audit:
                        audit.log_oauth(
                            workspace_id=workspace_id,
                            action="token_refresh",
                            success=False,
                            error="Tokens revoked by user",
                        )

                except ImportError:
                    logger.warning("Could not deactivate workspace - store unavailable")

        # Acknowledge the event
        return json_response({"ok": True})


# Handler factory function for registration
def create_slack_oauth_handler(server_context: Any) -> SlackOAuthHandler:
    """Factory function for handler registration."""
    return SlackOAuthHandler(server_context)
