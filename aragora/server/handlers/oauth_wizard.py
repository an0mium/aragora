"""
Unified OAuth Wizard Handler for SME Onboarding.

Provides a single API for discovering, configuring, and managing all platform
integrations through a unified wizard interface.

Endpoints:
- GET  /api/v2/integrations/wizard              - Get wizard configuration
- GET  /api/v2/integrations/wizard/providers    - List all available providers
- GET  /api/v2/integrations/wizard/status       - Get status of all integrations
- POST /api/v2/integrations/wizard/validate     - Validate configuration before connecting

This handler simplifies the onboarding experience for SMEs by providing:
1. Discovery of available integrations
2. Configuration validation
3. Pre-flight checks
4. Unified status overview
"""

from __future__ import annotations

import logging
import os
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

# Provider configurations
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "slack": {
        "name": "Slack",
        "description": "Connect Aragora to Slack workspaces for AI-powered debates in your channels",
        "category": "communication",
        "setup_time_minutes": 5,
        "features": [
            "Send debate results to channels",
            "Interactive slash commands",
            "Thread-based discussions",
            "Scheduled digests",
        ],
        "required_env_vars": ["SLACK_CLIENT_ID", "SLACK_CLIENT_SECRET"],
        "optional_env_vars": ["SLACK_REDIRECT_URI", "SLACK_SCOPES"],
        "oauth_scopes": [
            "channels:history",
            "chat:write",
            "commands",
            "users:read",
            "team:read",
            "channels:read",
        ],
        "install_url": "/api/integrations/slack/install",
        "docs_url": "https://docs.aragora.ai/integrations/slack",
    },
    "teams": {
        "name": "Microsoft Teams",
        "description": "Connect Aragora to Microsoft Teams for enterprise collaboration",
        "category": "communication",
        "setup_time_minutes": 10,
        "features": [
            "Send debate results to channels",
            "Adaptive Cards for rich content",
            "Tab integration",
            "Bot messaging",
        ],
        "required_env_vars": ["TEAMS_CLIENT_ID", "TEAMS_CLIENT_SECRET"],
        "optional_env_vars": ["TEAMS_REDIRECT_URI", "TEAMS_SCOPES"],
        "oauth_scopes": ["https://graph.microsoft.com/.default", "offline_access"],
        "install_url": "/api/integrations/teams/install",
        "docs_url": "https://docs.aragora.ai/integrations/teams",
    },
    "discord": {
        "name": "Discord",
        "description": "Connect Aragora to Discord servers for community engagement",
        "category": "communication",
        "setup_time_minutes": 5,
        "features": [
            "Bot commands",
            "Embed messages",
            "Server invites",
            "Role-based access",
        ],
        "required_env_vars": ["DISCORD_BOT_TOKEN"],
        "optional_env_vars": ["DISCORD_CLIENT_ID", "DISCORD_CLIENT_SECRET"],
        "oauth_scopes": ["bot", "applications.commands"],
        "install_url": "/api/integrations/discord/install",
        "docs_url": "https://docs.aragora.ai/integrations/discord",
    },
    "email": {
        "name": "Email (SMTP)",
        "description": "Send debate results and notifications via email",
        "category": "communication",
        "setup_time_minutes": 3,
        "features": [
            "HTML and plain text emails",
            "Scheduled digests",
            "Team notifications",
            "Custom templates",
        ],
        "required_env_vars": ["SMTP_HOST"],
        "optional_env_vars": ["SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"],
        "oauth_scopes": [],
        "install_url": None,  # No OAuth, direct configuration
        "docs_url": "https://docs.aragora.ai/integrations/email",
    },
    "gmail": {
        "name": "Gmail",
        "description": "Connect to Gmail for email integration with OAuth",
        "category": "communication",
        "setup_time_minutes": 5,
        "features": [
            "Send emails via Gmail",
            "Read inbox for triggers",
            "OAuth authentication",
            "No SMTP configuration needed",
        ],
        "required_env_vars": ["GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET"],
        "optional_env_vars": ["GOOGLE_OAUTH_REDIRECT_URI"],
        "oauth_scopes": [
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
        "install_url": "/api/integrations/gmail/install",
        "docs_url": "https://docs.aragora.ai/integrations/gmail",
    },
    "github": {
        "name": "GitHub",
        "description": "Connect to GitHub for code review debates and PR automation",
        "category": "development",
        "setup_time_minutes": 5,
        "features": [
            "PR review debates",
            "Issue triage",
            "Code analysis",
            "Automated comments",
        ],
        "required_env_vars": ["GITHUB_APP_ID", "GITHUB_PRIVATE_KEY"],
        "optional_env_vars": ["GITHUB_WEBHOOK_SECRET"],
        "oauth_scopes": ["repo", "read:org", "write:discussion"],
        "install_url": "/api/integrations/github/install",
        "docs_url": "https://docs.aragora.ai/integrations/github",
    },
}


class OAuthWizardHandler(BaseHandler):
    """
    Unified OAuth wizard handler for SME onboarding.

    Provides a single API for discovering and configuring all integrations.
    """

    ROUTES = [
        "/api/v2/integrations/wizard",
        "/api/v2/integrations/wizard/*",
    ]

    def __init__(self, server_context: ServerContext):
        """Initialize with server context."""
        super().__init__(server_context)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the request."""
        return path.startswith("/api/v2/integrations/wizard")

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

        try:
            # Main wizard endpoint
            if path == "/api/v2/integrations/wizard" and method == "GET":
                return await self._get_wizard_config(query_params)

            # List providers
            if path == "/api/v2/integrations/wizard/providers" and method == "GET":
                return await self._list_providers(query_params)

            # Get overall status
            if path == "/api/v2/integrations/wizard/status" and method == "GET":
                return await self._get_status(query_params)

            # Validate configuration
            if path == "/api/v2/integrations/wizard/validate" and method == "POST":
                return await self._validate_config(body)

            return error_response("Not found", 404)

        except Exception as e:
            logger.exception(f"Error in OAuth wizard handler: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    async def _get_wizard_config(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Get the complete wizard configuration.

        Returns all provider information, configuration status, and next steps.
        """
        providers_status: List[Dict[str, Any]] = []

        for provider_id, provider in PROVIDERS.items():
            status = self._check_provider_config(provider_id, provider)
            providers_status.append(
                {
                    "id": provider_id,
                    **provider,
                    "status": status,
                }
            )

        # Sort by category and configuration status
        providers_status.sort(key=lambda p: (p["category"], not p["status"]["configured"]))

        # Calculate overall readiness
        configured_count = sum(1 for p in providers_status if p["status"]["configured"])
        total_count = len(providers_status)

        return json_response(
            {
                "wizard": {
                    "version": "1.0",
                    "providers": providers_status,
                    "summary": {
                        "total_providers": total_count,
                        "configured": configured_count,
                        "ready_to_use": sum(
                            1
                            for p in providers_status
                            if p["status"]["configured"] and not p["status"]["errors"]
                        ),
                    },
                    "recommended_order": [
                        "slack",  # Most common for SMEs
                        "teams",  # Enterprise alternative
                        "email",  # Easiest to configure
                        "github",  # For dev teams
                        "discord",  # For communities
                        "gmail",  # Alternative email
                    ],
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _list_providers(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        List all available providers.

        Query params:
            category: Filter by category (communication, development)
            configured: Filter by configuration status (true, false)
        """
        filter_category = query_params.get("category")
        filter_configured = query_params.get("configured")

        providers_list = []

        for provider_id, provider in PROVIDERS.items():
            # Apply category filter
            if filter_category and provider["category"] != filter_category:
                continue

            status = self._check_provider_config(provider_id, provider)

            # Apply configured filter
            if filter_configured is not None:
                is_configured = filter_configured.lower() == "true"
                if status["configured"] != is_configured:
                    continue

            providers_list.append(
                {
                    "id": provider_id,
                    "name": provider["name"],
                    "description": provider["description"],
                    "category": provider["category"],
                    "setup_time_minutes": provider["setup_time_minutes"],
                    "features": provider["features"],
                    "configured": status["configured"],
                    "install_url": provider["install_url"],
                    "docs_url": provider["docs_url"],
                }
            )

        return json_response(
            {
                "providers": providers_list,
                "total": len(providers_list),
            }
        )

    async def _get_status(self, query_params: Dict[str, str]) -> HandlerResult:
        """
        Get detailed status of all integrations.

        Includes configuration status, health checks, and connection details.
        """
        statuses: List[Dict[str, Any]] = []

        for provider_id, provider in PROVIDERS.items():
            config_status = self._check_provider_config(provider_id, provider)

            # Get connection status if configured
            connection_status = None
            if config_status["configured"]:
                connection_status = await self._check_connection(provider_id)

            statuses.append(
                {
                    "provider_id": provider_id,
                    "name": provider["name"],
                    "category": provider["category"],
                    "configuration": config_status,
                    "connection": connection_status,
                }
            )

        # Summary
        configured = sum(1 for s in statuses if s["configuration"]["configured"])
        connected = sum(
            1 for s in statuses if s["connection"] and s["connection"].get("status") == "connected"
        )

        return json_response(
            {
                "statuses": statuses,
                "summary": {
                    "total": len(statuses),
                    "configured": configured,
                    "connected": connected,
                    "needs_attention": sum(
                        1
                        for s in statuses
                        if s["configuration"]["errors"]
                        or (s["connection"] and s["connection"].get("status") == "error")
                    ),
                },
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def _validate_config(self, body: Dict[str, Any]) -> HandlerResult:
        """
        Validate configuration for a provider before connecting.

        Body:
            provider: Provider ID to validate
            config: Optional configuration values to validate
        """
        provider_id = body.get("provider")
        config = body.get("config", {})

        if not provider_id:
            return error_response("Provider ID is required", 400)

        if provider_id not in PROVIDERS:
            return error_response(
                f"Unknown provider: {provider_id}. Available: {', '.join(PROVIDERS.keys())}",
                400,
            )

        provider = PROVIDERS[provider_id]
        validation_results: Dict[str, Any] = {
            "provider": provider_id,
            "valid": True,
            "checks": [],
        }

        # Check required environment variables
        for env_var in provider["required_env_vars"]:
            value = config.get(env_var) or os.environ.get(env_var)
            check = {
                "name": env_var,
                "type": "env_var",
                "required": True,
                "present": bool(value),
            }
            if not value:
                check["error"] = f"Missing required environment variable: {env_var}"
                validation_results["valid"] = False
            validation_results["checks"].append(check)

        # Check optional environment variables
        for env_var in provider["optional_env_vars"]:
            value = config.get(env_var) or os.environ.get(env_var)
            validation_results["checks"].append(
                {
                    "name": env_var,
                    "type": "env_var",
                    "required": False,
                    "present": bool(value),
                }
            )

        # Add recommendations
        validation_results["recommendations"] = []
        if validation_results["valid"]:
            validation_results["recommendations"].append(
                f"Configuration looks good! Visit {provider['install_url']} to complete setup."
            )
        else:
            missing = [
                c["name"]
                for c in validation_results["checks"]
                if c["required"] and not c["present"]
            ]
            validation_results["recommendations"].append(
                f"Set the following environment variables: {', '.join(missing)}"
            )
            validation_results["recommendations"].append(
                f"See {provider['docs_url']} for detailed setup instructions."
            )

        return json_response(validation_results)

    def _check_provider_config(self, provider_id: str, provider: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a provider is properly configured."""
        errors: List[str] = []
        warnings: List[str] = []

        # Check required env vars
        missing_required = []
        for env_var in provider["required_env_vars"]:
            if not os.environ.get(env_var):
                missing_required.append(env_var)

        if missing_required:
            errors.append(f"Missing required: {', '.join(missing_required)}")

        # Check optional env vars
        missing_optional = []
        for env_var in provider["optional_env_vars"]:
            if not os.environ.get(env_var):
                missing_optional.append(env_var)

        if missing_optional:
            warnings.append(f"Optional not set: {', '.join(missing_optional)}")

        return {
            "configured": len(missing_required) == 0,
            "errors": errors,
            "warnings": warnings,
            "required_vars_present": len(provider["required_env_vars"]) - len(missing_required),
            "required_vars_total": len(provider["required_env_vars"]),
        }

    async def _check_connection(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Check the connection status for a configured provider."""
        try:
            if provider_id == "slack":
                return await self._check_slack_connection()
            elif provider_id == "teams":
                return await self._check_teams_connection()
            elif provider_id == "discord":
                return await self._check_discord_connection()
            elif provider_id == "email":
                return await self._check_email_connection()
            else:
                return {"status": "unchecked", "reason": "No health check available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_slack_connection(self) -> Dict[str, Any]:
        """Check Slack connection status."""
        try:
            from aragora.storage.slack_workspace_store import get_slack_workspace_store

            store = get_slack_workspace_store()
            workspaces = store.list_active(limit=1)

            if workspaces:
                return {
                    "status": "connected",
                    "workspaces": len(store.list_active(limit=100)),
                }
            else:
                return {"status": "not_connected", "reason": "No active workspaces"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_teams_connection(self) -> Dict[str, Any]:
        """Check Teams connection status."""
        try:
            from aragora.storage.teams_workspace_store import get_teams_workspace_store

            store = get_teams_workspace_store()
            workspaces = store.list_active(limit=1)

            if workspaces:
                return {
                    "status": "connected",
                    "tenants": len(store.list_active(limit=100)),
                }
            else:
                return {"status": "not_connected", "reason": "No active tenants"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _check_discord_connection(self) -> Dict[str, Any]:
        """Check Discord connection status."""
        bot_token = os.environ.get("DISCORD_BOT_TOKEN")
        if not bot_token:
            return {"status": "not_configured"}

        return {"status": "configured", "note": "Bot token present"}

    async def _check_email_connection(self) -> Dict[str, Any]:
        """Check email/SMTP connection status."""
        import socket

        smtp_host = os.environ.get("SMTP_HOST")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))

        if not smtp_host:
            return {"status": "not_configured"}

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((smtp_host, smtp_port))
            sock.close()

            if result == 0:
                return {"status": "connected", "smtp_host": smtp_host, "smtp_port": smtp_port}
            else:
                return {"status": "unreachable", "smtp_host": smtp_host, "smtp_port": smtp_port}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Handler factory function for registration
def create_oauth_wizard_handler(server_context: ServerContext) -> OAuthWizardHandler:
    """Factory function for handler registration."""
    return OAuthWizardHandler(server_context)
