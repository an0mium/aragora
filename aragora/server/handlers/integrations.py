"""
Integration Management HTTP Handlers for Aragora.

Provides REST API endpoints for managing platform integrations:
- List connected integrations for a workspace/tenant
- Disconnect integrations
- Get integration status and health
- Manage integration settings

Endpoints:
    GET  /api/v2/integrations                    - List all integrations
    GET  /api/v2/integrations/:type              - Get specific integration status
    DELETE /api/v2/integrations/:type            - Disconnect integration
    POST /api/v2/integrations/:type/test         - Test integration connectivity
    GET  /api/v2/integrations/stats              - Integration statistics

Supports Slack, Teams, Discord, and Email integrations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    ServerContext,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Supported integration types
SUPPORTED_INTEGRATIONS = {"slack", "teams", "discord", "email"}


class IntegrationsHandler(BaseHandler):
    """
    HTTP handler for managing platform integrations.

    Provides REST API access to view, manage, and test integrations
    with external platforms like Slack, Teams, Discord, and Email.
    """

    ROUTES = [
        "/api/v2/integrations",
        "/api/v2/integrations/*",
    ]

    def __init__(self, server_context: ServerContext):
        """Initialize with server context."""
        super().__init__(server_context)
        self._slack_store = None
        self._teams_store = None

    def _get_slack_store(self):
        """Get or create Slack workspace store (lazy initialization)."""
        if self._slack_store is None:
            from aragora.storage.slack_workspace_store import get_slack_workspace_store

            self._slack_store = get_slack_workspace_store()
        return self._slack_store

    def _get_teams_store(self):
        """Get or create Teams workspace store (lazy initialization)."""
        if self._teams_store is None:
            from aragora.storage.teams_workspace_store import get_teams_workspace_store

            self._teams_store = get_teams_workspace_store()
        return self._teams_store

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the request."""
        if path.startswith("/api/v2/integrations"):
            return method in ("GET", "POST", "DELETE")
        return False

    @rate_limit(requests_per_minute=60)
    async def handle(  # type: ignore[override]
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HandlerResult:
        """Route request to appropriate handler method."""
        query_params = query_params or {}
        body = body or {}

        # Extract tenant ID from auth context (header or query param)
        tenant_id = (headers.get("X-Tenant-ID") if headers else None) or query_params.get(
            "tenant_id"
        )

        try:
            # Stats endpoint
            if path == "/api/v2/integrations/stats" and method == "GET":
                return await self._get_stats(tenant_id)

            # List all integrations
            if path == "/api/v2/integrations" and method == "GET":
                return await self._list_integrations(tenant_id, query_params)

            # Integration-specific routes
            if path.startswith("/api/v2/integrations/"):
                parts = path.split("/")
                if len(parts) < 5:
                    return error_response("Invalid integration path", 400)

                integration_type = parts[4]

                if integration_type not in SUPPORTED_INTEGRATIONS:
                    return error_response(
                        f"Unknown integration: {integration_type}. "
                        f"Supported: {', '.join(sorted(SUPPORTED_INTEGRATIONS))}",
                        400,
                    )

                # Health endpoint (GET)
                if len(parts) > 5 and parts[5] == "health" and method == "GET":
                    workspace_id = query_params.get("workspace_id")
                    return await self._get_health(integration_type, workspace_id, tenant_id)

                # Test endpoint (POST - legacy, calls health check)
                if len(parts) > 5 and parts[5] == "test" and method == "POST":
                    workspace_id = body.get("workspace_id") or query_params.get("workspace_id")
                    return await self._test_integration(integration_type, workspace_id, tenant_id)

                # Get specific integration
                if method == "GET":
                    workspace_id = query_params.get("workspace_id")
                    return await self._get_integration(integration_type, workspace_id, tenant_id)

                # Disconnect integration
                if method == "DELETE":
                    workspace_id = body.get("workspace_id") or query_params.get("workspace_id")
                    return await self._disconnect_integration(
                        integration_type, workspace_id, tenant_id
                    )

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error handling integration request: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    async def _list_integrations(
        self, tenant_id: Optional[str], query_params: Dict[str, str]
    ) -> HandlerResult:
        """
        List all integrations for a tenant.

        Query params:
            limit: Max results (default 20, max 100)
            offset: Pagination offset
            type: Filter by integration type
            status: Filter by status (active, inactive)
        """
        limit = min(int(query_params.get("limit", "20")), 100)
        offset = int(query_params.get("offset", "0"))
        filter_type = query_params.get("type")
        filter_status = query_params.get("status")

        integrations: List[Dict[str, Any]] = []

        # Get Slack integrations
        if not filter_type or filter_type == "slack":
            slack_store = self._get_slack_store()
            if tenant_id:
                slack_workspaces = slack_store.get_by_tenant(tenant_id)
            else:
                slack_workspaces = slack_store.list_active(limit=limit, offset=offset)

            for ws in slack_workspaces:
                if filter_status:
                    if filter_status == "active" and not ws.is_active:
                        continue
                    if filter_status == "inactive" and ws.is_active:
                        continue

                integrations.append(
                    {
                        "type": "slack",
                        "workspace_id": ws.workspace_id,
                        "workspace_name": ws.workspace_name,
                        "status": "active" if ws.is_active else "inactive",
                        "installed_at": ws.installed_at,
                        "installed_by": ws.installed_by,
                        "scopes": ws.scopes,
                        "has_refresh_token": bool(ws.refresh_token),
                        "token_expires_at": ws.token_expires_at,
                    }
                )

        # Get Teams integrations
        if not filter_type or filter_type == "teams":
            teams_store = self._get_teams_store()
            if tenant_id:
                teams_workspaces = teams_store.get_by_aragora_tenant(tenant_id)
            else:
                teams_workspaces = teams_store.list_active(limit=limit, offset=offset)

            for ws in teams_workspaces:
                if filter_status:
                    if filter_status == "active" and not ws.is_active:
                        continue
                    if filter_status == "inactive" and ws.is_active:
                        continue

                integrations.append(
                    {
                        "type": "teams",
                        "tenant_id": ws.tenant_id,
                        "tenant_name": ws.tenant_name,
                        "status": "active" if ws.is_active else "inactive",
                        "installed_at": ws.installed_at,
                        "installed_by": ws.installed_by,
                        "scopes": ws.scopes,
                        "has_refresh_token": bool(ws.refresh_token),
                        "token_expires_at": ws.token_expires_at,
                    }
                )

        # Apply pagination
        total = len(integrations)
        integrations = integrations[offset : offset + limit]

        return json_response(
            {
                "integrations": integrations,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": total,
                    "has_more": offset + len(integrations) < total,
                },
            }
        )

    async def _get_integration(
        self,
        integration_type: str,
        workspace_id: Optional[str],
        tenant_id: Optional[str],
    ) -> HandlerResult:
        """Get specific integration details."""
        if integration_type == "slack":
            store = self._get_slack_store()

            if workspace_id:
                workspace = store.get(workspace_id)
                if not workspace:
                    return error_response("Slack workspace not found", 404)

                return json_response(
                    {
                        "type": "slack",
                        "connected": True,
                        "workspace": workspace.to_dict(),
                        "health": await self._check_slack_health(workspace),
                    }
                )

            # List all for tenant
            workspaces = (
                store.get_by_tenant(tenant_id) if tenant_id else store.list_active(limit=10)
            )

            return json_response(
                {
                    "type": "slack",
                    "connected": len(workspaces) > 0,
                    "workspaces": [ws.to_dict() for ws in workspaces],
                    "count": len(workspaces),
                }
            )

        elif integration_type == "teams":
            store = self._get_teams_store()

            if workspace_id:
                workspace = store.get(workspace_id)
                if not workspace:
                    return error_response("Teams tenant not found", 404)

                return json_response(
                    {
                        "type": "teams",
                        "connected": True,
                        "workspace": workspace.to_dict(),
                        "health": await self._check_teams_health(workspace),
                    }
                )

            # List all for tenant
            workspaces = (
                store.get_by_aragora_tenant(tenant_id) if tenant_id else store.list_active(limit=10)
            )

            return json_response(
                {
                    "type": "teams",
                    "connected": len(workspaces) > 0,
                    "workspaces": [ws.to_dict() for ws in workspaces],
                    "count": len(workspaces),
                }
            )

        elif integration_type == "discord":
            import os

            has_token = bool(os.environ.get("DISCORD_BOT_TOKEN"))
            return json_response(
                {
                    "type": "discord",
                    "connected": has_token,
                    "configured": has_token,
                    "note": "Discord uses bot token authentication",
                }
            )

        elif integration_type == "email":
            import os

            smtp_configured = bool(os.environ.get("SMTP_HOST"))
            return json_response(
                {
                    "type": "email",
                    "connected": smtp_configured,
                    "configured": smtp_configured,
                    "smtp_host": os.environ.get("SMTP_HOST", "not configured"),
                }
            )

        return error_response(f"Unknown integration type: {integration_type}", 400)

    async def _disconnect_integration(
        self,
        integration_type: str,
        workspace_id: Optional[str],
        tenant_id: Optional[str],
    ) -> HandlerResult:
        """Disconnect an integration."""
        if not workspace_id:
            return error_response("workspace_id is required", 400)

        if integration_type == "slack":
            store = self._get_slack_store()
            workspace = store.get(workspace_id)

            if not workspace:
                return error_response("Slack workspace not found", 404)

            # Deactivate (soft delete)
            success = store.deactivate(workspace_id)

            if success:
                logger.info(f"Disconnected Slack workspace: {workspace_id}")
                return json_response(
                    {
                        "disconnected": True,
                        "type": "slack",
                        "workspace_id": workspace_id,
                        "workspace_name": workspace.workspace_name,
                    }
                )

            return error_response("Failed to disconnect Slack workspace", 500)

        elif integration_type == "teams":
            store = self._get_teams_store()
            workspace = store.get(workspace_id)

            if not workspace:
                return error_response("Teams tenant not found", 404)

            success = store.deactivate(workspace_id)

            if success:
                logger.info(f"Disconnected Teams tenant: {workspace_id}")
                return json_response(
                    {
                        "disconnected": True,
                        "type": "teams",
                        "tenant_id": workspace_id,
                        "tenant_name": workspace.tenant_name,
                    }
                )

            return error_response("Failed to disconnect Teams tenant", 500)

        return error_response(f"Cannot disconnect {integration_type}: not supported", 400)

    async def _test_integration(
        self,
        integration_type: str,
        workspace_id: Optional[str],
        tenant_id: Optional[str],
    ) -> HandlerResult:
        """Test integration connectivity."""
        if integration_type == "slack":
            if not workspace_id:
                return error_response("workspace_id is required for Slack test", 400)

            store = self._get_slack_store()
            workspace = store.get(workspace_id)

            if not workspace:
                return error_response("Slack workspace not found", 404)

            health = await self._check_slack_health(workspace)
            return json_response(
                {
                    "type": "slack",
                    "workspace_id": workspace_id,
                    "test_result": health,
                    "tested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        elif integration_type == "teams":
            if not workspace_id:
                return error_response("workspace_id is required for Teams test", 400)

            store = self._get_teams_store()
            workspace = store.get(workspace_id)

            if not workspace:
                return error_response("Teams tenant not found", 404)

            health = await self._check_teams_health(workspace)
            return json_response(
                {
                    "type": "teams",
                    "tenant_id": workspace_id,
                    "test_result": health,
                    "tested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        elif integration_type == "discord":
            health = await self._check_discord_health()
            return json_response(
                {
                    "type": "discord",
                    "test_result": health,
                    "tested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        elif integration_type == "email":
            health = await self._check_email_health()
            return json_response(
                {
                    "type": "email",
                    "test_result": health,
                    "tested_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        return error_response(f"Cannot test {integration_type}", 400)

    async def _get_health(
        self,
        integration_type: str,
        workspace_id: Optional[str],
        tenant_id: Optional[str],
    ) -> HandlerResult:
        """
        Get health status for an integration.

        Returns detailed health information including:
        - Connection status
        - Token validity
        - Last successful operation
        - Error details if unhealthy
        """
        if integration_type == "slack":
            if not workspace_id:
                # Return aggregate health for all Slack workspaces
                store = self._get_slack_store()
                workspaces = (
                    store.get_by_tenant(tenant_id) if tenant_id else store.list_active(limit=10)
                )

                if not workspaces:
                    return json_response(
                        {
                            "type": "slack",
                            "status": "not_configured",
                            "healthy": False,
                            "workspaces": [],
                        }
                    )

                workspace_health = []
                all_healthy = True
                for ws in workspaces:
                    health = await self._check_slack_health(ws)
                    is_healthy = health.get("status") == "healthy"
                    all_healthy = all_healthy and is_healthy
                    workspace_health.append(
                        {
                            "workspace_id": ws.workspace_id,
                            "workspace_name": ws.workspace_name,
                            **health,
                        }
                    )

                return json_response(
                    {
                        "type": "slack",
                        "status": "healthy" if all_healthy else "degraded",
                        "healthy": all_healthy,
                        "workspaces": workspace_health,
                    }
                )

            # Health for specific workspace
            store = self._get_slack_store()
            workspace = store.get(workspace_id)
            if not workspace:
                return error_response("Slack workspace not found", 404)

            health = await self._check_slack_health(workspace)
            return json_response(
                {
                    "type": "slack",
                    "workspace_id": workspace_id,
                    "workspace_name": workspace.workspace_name,
                    "healthy": health.get("status") == "healthy",
                    **health,
                }
            )

        elif integration_type == "teams":
            if not workspace_id:
                store = self._get_teams_store()
                workspaces = (
                    store.get_by_aragora_tenant(tenant_id)
                    if tenant_id
                    else store.list_active(limit=10)
                )

                if not workspaces:
                    return json_response(
                        {
                            "type": "teams",
                            "status": "not_configured",
                            "healthy": False,
                            "workspaces": [],
                        }
                    )

                workspace_health = []
                all_healthy = True
                for ws in workspaces:
                    health = await self._check_teams_health(ws)
                    is_healthy = health.get("status") == "healthy"
                    all_healthy = all_healthy and is_healthy
                    workspace_health.append(
                        {"tenant_id": ws.tenant_id, "tenant_name": ws.tenant_name, **health}
                    )

                return json_response(
                    {
                        "type": "teams",
                        "status": "healthy" if all_healthy else "degraded",
                        "healthy": all_healthy,
                        "workspaces": workspace_health,
                    }
                )

            store = self._get_teams_store()
            workspace = store.get(workspace_id)
            if not workspace:
                return error_response("Teams tenant not found", 404)

            health = await self._check_teams_health(workspace)
            return json_response(
                {
                    "type": "teams",
                    "tenant_id": workspace_id,
                    "tenant_name": workspace.tenant_name,
                    "healthy": health.get("status") == "healthy",
                    **health,
                }
            )

        elif integration_type == "discord":
            health = await self._check_discord_health()
            return json_response(
                {"type": "discord", "healthy": health.get("status") == "healthy", **health}
            )

        elif integration_type == "email":
            health = await self._check_email_health()
            return json_response(
                {"type": "email", "healthy": health.get("status") == "healthy", **health}
            )

        return error_response(f"Unknown integration type: {integration_type}", 400)

    async def _get_stats(self, tenant_id: Optional[str]) -> HandlerResult:
        """Get integration statistics."""
        slack_store = self._get_slack_store()
        teams_store = self._get_teams_store()

        slack_stats = slack_store.get_stats()
        teams_stats = teams_store.get_stats()

        return json_response(
            {
                "stats": {
                    "slack": slack_stats,
                    "teams": teams_stats,
                    "total_integrations": (
                        slack_stats.get("active_workspaces", 0)
                        + teams_stats.get("active_workspaces", 0)
                    ),
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _check_slack_health(self, workspace) -> Dict[str, Any]:
        """Check Slack workspace health."""
        try:
            import json
            import urllib.request

            request = urllib.request.Request(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {workspace.access_token}"},
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode())

            if result.get("ok"):
                return {
                    "status": "healthy",
                    "team": result.get("team"),
                    "bot_id": result.get("bot_id"),
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": result.get("error", "unknown"),
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_teams_health(self, workspace) -> Dict[str, Any]:
        """Check Teams workspace health."""
        try:
            import json
            import urllib.request

            # Test Graph API access
            request = urllib.request.Request(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {workspace.access_token}"},
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode())

            return {
                "status": "healthy",
                "display_name": result.get("displayName"),
            }

        except urllib.error.HTTPError as e:
            if e.code == 401:
                return {"status": "token_expired", "error": "Token needs refresh"}
            return {"status": "unhealthy", "error": str(e)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_discord_health(self) -> Dict[str, Any]:
        """Check Discord bot health."""
        import os

        bot_token = os.environ.get("DISCORD_BOT_TOKEN")
        if not bot_token:
            return {"status": "not_configured", "error": "No bot token"}

        try:
            import json
            import urllib.request

            request = urllib.request.Request(
                "https://discord.com/api/v10/users/@me",
                headers={"Authorization": f"Bot {bot_token}"},
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode())

            return {
                "status": "healthy",
                "bot_name": result.get("username"),
                "bot_id": result.get("id"),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_email_health(self) -> Dict[str, Any]:
        """Check email/SMTP health."""
        import os
        import socket

        smtp_host = os.environ.get("SMTP_HOST")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))

        if not smtp_host:
            return {"status": "not_configured", "error": "No SMTP host"}

        try:
            # Test TCP connection to SMTP
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((smtp_host, smtp_port))
            sock.close()

            if result == 0:
                return {
                    "status": "healthy",
                    "smtp_host": smtp_host,
                    "smtp_port": smtp_port,
                }
            else:
                return {
                    "status": "unreachable",
                    "error": f"Cannot connect to {smtp_host}:{smtp_port}",
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}


# Handler factory function for registration
def create_integrations_handler(server_context: ServerContext) -> IntegrationsHandler:
    """Factory function for handler registration."""
    return IntegrationsHandler(server_context)
