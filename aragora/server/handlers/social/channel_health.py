"""
Channel Health Handler - Aggregated health status for all chat channels.

Provides endpoints for monitoring the health of chat platform integrations
including Slack, Teams, Discord, Telegram, WhatsApp, and Email.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from aiohttp import web

from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)


class ChannelHealthHandler:
    """
    Handler for chat channel health monitoring.

    Provides:
    - GET /api/v1/channels/health - Aggregated health of all channels
    - GET /api/v1/channels/{channel}/health - Health of specific channel
    """

    ROUTES = [
        ("GET", "/api/v1/channels/health", "get_all_channels_health"),
        ("GET", "/api/v1/channels/{channel}/health", "get_channel_health"),
    ]

    def __init__(self) -> None:
        self._connectors: dict[str, Any] = {}
        self._initialized = False

    def _ensure_connectors_initialized(self) -> None:
        """Lazy initialization of channel connectors."""
        if self._initialized:
            return

        # Import connectors lazily to avoid circular imports
        try:
            from aragora.connectors.chat.slack import SlackConnector

            self._connectors["slack"] = SlackConnector()
        except Exception as e:
            logger.debug(f"Slack connector not available: {e}")

        try:
            from aragora.connectors.chat.teams import TeamsConnector

            self._connectors["teams"] = TeamsConnector()
        except Exception as e:
            logger.debug(f"Teams connector not available: {e}")

        try:
            from aragora.connectors.chat.discord import DiscordConnector

            self._connectors["discord"] = DiscordConnector()
        except Exception as e:
            logger.debug(f"Discord connector not available: {e}")

        try:
            from aragora.connectors.chat.telegram import TelegramConnector

            self._connectors["telegram"] = TelegramConnector()
        except Exception as e:
            logger.debug(f"Telegram connector not available: {e}")

        try:
            from aragora.connectors.chat.whatsapp import WhatsAppConnector

            self._connectors["whatsapp"] = WhatsAppConnector()
        except Exception as e:
            logger.debug(f"WhatsApp connector not available: {e}")

        try:
            from aragora.connectors.chat.google_chat import GoogleChatConnector

            self._connectors["google_chat"] = GoogleChatConnector()
        except Exception as e:
            logger.debug(f"Google Chat connector not available: {e}")

        # Check for email integration
        try:
            from aragora.integrations.email import EmailConfig, EmailIntegration

            self._connectors["email"] = EmailIntegration(EmailConfig())
        except Exception as e:
            logger.debug(f"Email integration not available: {e}")

        self._initialized = True

    async def get_all_channels_health(self, request: web.Request) -> web.Response:
        """
        Get aggregated health status for all chat channels.

        Returns:
            JSON response with health status for each channel and summary.
        """
        self._ensure_connectors_initialized()

        channels: dict[str, Any] = {}
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        unconfigured_count = 0

        for name, connector in self._connectors.items():
            try:
                if hasattr(connector, "get_health"):
                    health = await connector.get_health()
                else:
                    # Fallback for connectors without get_health
                    health = await self._get_basic_health(name, connector)

                channels[name] = health
                status = health.get("status", "unknown")

                if status == "healthy":
                    healthy_count += 1
                elif status == "degraded":
                    degraded_count += 1
                elif status == "unhealthy":
                    unhealthy_count += 1
                elif status == "unconfigured":
                    unconfigured_count += 1

            except Exception as e:
                logger.error(f"Error getting health for {name}: {e}")
                channels[name] = {
                    "platform": name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                }
                unhealthy_count += 1

        # Determine overall status
        total_configured = healthy_count + degraded_count + unhealthy_count
        if total_configured == 0:
            overall_status = "no_channels_configured"
        elif unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Calculate health score (0-100)
        if total_configured > 0:
            health_score = int((healthy_count * 100 + degraded_count * 50) / total_configured)
        else:
            health_score = 0

        response = {
            "status": overall_status,
            "health_score": health_score,
            "timestamp": time.time(),
            "summary": {
                "total": len(self._connectors),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "unconfigured": unconfigured_count,
            },
            "channels": channels,
        }

        return web.json_response(response)

    async def get_channel_health(self, request: web.Request) -> web.Response:
        """
        Get health status for a specific channel.

        Args:
            request: HTTP request with 'channel' path parameter

        Returns:
            JSON response with channel health status.
        """
        self._ensure_connectors_initialized()

        channel = request.match_info.get("channel", "").lower()

        if channel not in self._connectors:
            return web.json_response(
                {
                    "error": f"Unknown channel: {channel}",
                    "available_channels": list(self._connectors.keys()),
                },
                status=404,
            )

        connector = self._connectors[channel]

        try:
            if hasattr(connector, "get_health"):
                health = await connector.get_health()
            else:
                health = await self._get_basic_health(channel, connector)

            return web.json_response(health)

        except Exception as e:
            logger.error(f"Error getting health for {channel}: {e}")
            return web.json_response(
                {
                    "platform": channel,
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                },
                status=500,
            )

    async def _get_basic_health(self, name: str, connector: Any) -> dict[str, Any]:
        """Get basic health info for connectors without get_health method."""
        health: dict[str, Any] = {
            "platform": name,
            "status": "unknown",
            "timestamp": time.time(),
            "details": {},
        }

        # Check if connector has is_configured property
        if hasattr(connector, "is_configured"):
            configured = connector.is_configured
            health["configured"] = configured
            if not configured:
                health["status"] = "unconfigured"
                return health

        # Check for test_connection method
        if hasattr(connector, "test_connection"):
            try:
                result = await connector.test_connection()
                health["status"] = "healthy" if result.get("success") else "unhealthy"
                health["details"] = result
            except Exception as e:
                health["status"] = "unhealthy"
                health["details"]["error"] = str(e)
        else:
            # Assume healthy if configured and no test method
            if health.get("configured", True):
                health["status"] = "healthy"

        return health

    def can_handle(self, method: str, path: str) -> bool:
        """Check if this handler can handle the request."""
        if not path.startswith("/api/v1/channels"):
            return False

        # Normalize path
        normalized = path.rstrip("/")

        # Check exact matches
        if method == "GET" and normalized == "/api/v1/channels/health":
            return True

        # Check pattern match for specific channel health
        # Path: /api/v1/channels/{channel}/health
        # Parts: ['', 'api', 'v1', 'channels', '{channel}', 'health'] = 6 parts
        if method == "GET" and normalized.startswith("/api/v1/channels/"):
            parts = normalized.split("/")
            if len(parts) == 6 and parts[5] == "health":
                return True

        return False

    @require_permission("channels:read")
    async def handle(self, request: web.Request) -> web.Response:
        """Route the request to the appropriate handler method."""
        path = request.path.rstrip("/")
        method = request.method

        if method == "GET" and path == "/api/v1/channels/health":
            return await self.get_all_channels_health(request)

        if method == "GET" and path.startswith("/api/v1/channels/"):
            parts = path.split("/")
            if len(parts) == 6 and parts[5] == "health":
                return await self.get_channel_health(request)

        return web.json_response({"error": "Not found"}, status=404)
