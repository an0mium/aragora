"""
Integration API Handler.

Provides REST API endpoints for chat platform integration management:
- GET    /api/integrations/status       - Get status of all integrations
- GET    /api/integrations/:type        - Get configuration for specific integration
- PUT    /api/integrations/:type        - Configure/update integration
- PATCH  /api/integrations/:type        - Partial update (enable/disable)
- DELETE /api/integrations/:type        - Remove integration configuration
- POST   /api/integrations/:type/test   - Test integration connection

Supported integration types:
- slack, discord, telegram, email, teams, whatsapp, matrix

Storage:
- Integration configs are persisted to SQLite/Redis via IntegrationStore
- Survives server restarts and supports multi-instance deployments
"""

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

from aragora.server.handlers.base import (
    error_response,
    json_response,
)
from aragora.server.handlers.utils.responses import HandlerResult
from aragora.server.handlers.secure import SecureHandler
from aragora.storage.integration_store import (
    IntegrationConfig,
    VALID_INTEGRATION_TYPES,
    get_integration_store,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Integration Types and Models
# =============================================================================

IntegrationType = Literal["slack", "discord", "telegram", "email", "teams", "whatsapp", "matrix"]


@dataclass
class IntegrationStatus:
    """Status information for an integration."""

    type: IntegrationType
    enabled: bool
    status: str  # connected, degraded, disconnected, not_configured
    messages_sent: int = 0
    errors: int = 0
    last_activity: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Handler Class
# =============================================================================


class IntegrationsHandler(SecureHandler):
    """Handler for integration management endpoints.

    Extends SecureHandler for JWT-based authentication and audit logging.
    """

    RESOURCE_TYPE = "integration"

    """Handler for integration management endpoints."""

    async def get_status(self, user_id: str = "default") -> HandlerResult:
        """Get status of all integrations.

        Returns:
            Status for each integration type
        """
        store = get_integration_store()
        statuses: List[IntegrationStatus] = []

        for int_type in VALID_INTEGRATION_TYPES:
            config = await store.get(int_type, user_id)

            if config:
                status = IntegrationStatus(
                    type=config.type,  # type: ignore
                    enabled=config.enabled,
                    status=config.status,
                    messages_sent=config.messages_sent,
                    errors=config.errors_24h,
                    last_activity=(
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(config.last_activity))
                        if config.last_activity
                        else None
                    ),
                )
            else:
                status = IntegrationStatus(
                    type=int_type,  # type: ignore
                    enabled=False,
                    status="not_configured",
                )

            statuses.append(status)

        return json_response({"integrations": [s.to_dict() for s in statuses]})

    async def get_integration(
        self, integration_type: str, user_id: str = "default"
    ) -> HandlerResult:
        """Get configuration for specific integration.

        Args:
            integration_type: Type of integration (slack, discord, etc.)
            user_id: User/workspace ID

        Returns:
            Integration configuration
        """
        if integration_type not in VALID_INTEGRATION_TYPES:
            return error_response(f"Invalid integration type: {integration_type}", status=400)

        store = get_integration_store()
        config = await store.get(integration_type, user_id)

        if not config:
            return error_response(f"Integration not configured: {integration_type}", status=404)

        return json_response({"integration": config.to_dict()})

    async def configure_integration(
        self,
        integration_type: str,
        data: Dict[str, Any],
        user_id: str = "default",
    ) -> HandlerResult:
        """Configure or update an integration.

        Args:
            integration_type: Type of integration
            data: Configuration data
            user_id: User/workspace ID

        Returns:
            Updated configuration
        """
        if integration_type not in VALID_INTEGRATION_TYPES:
            return error_response(f"Invalid integration type: {integration_type}", status=400)

        store = get_integration_store()
        existing = await store.get(integration_type, user_id)

        # Extract notification settings
        notify_settings = {
            "notify_on_consensus": data.get("notify_on_consensus", True),
            "notify_on_debate_end": data.get("notify_on_debate_end", True),
            "notify_on_error": data.get("notify_on_error", False),
            "notify_on_leaderboard": data.get("notify_on_leaderboard", False),
        }

        # Extract provider-specific settings
        provider_keys = [
            "webhook_url",
            "bot_token",
            "channel",
            "chat_id",
            "access_token",
            "phone_number_id",
            "recipient",
            "homeserver_url",
            "room_id",
            "user_id",
            "provider",
            "from_email",
            "from_name",
            "smtp_host",
            "smtp_port",
            "smtp_username",
            "smtp_password",
            "sendgrid_api_key",
            "ses_region",
            "ses_access_key_id",
            "ses_secret_access_key",
            "twilio_account_sid",
            "twilio_auth_token",
            "twilio_whatsapp_number",
            "username",
            "avatar_url",
            "parse_mode",
            "use_html",
            "enable_commands",
            "use_adaptive_cards",
            "reply_to",
        ]
        settings = {k: data[k] for k in provider_keys if k in data}

        if existing:
            # Update existing
            existing.settings.update(settings)
            existing.updated_at = time.time()
            existing.enabled = data.get("enabled", existing.enabled)
            for k, v in notify_settings.items():
                setattr(existing, k, v)
            config = existing
        else:
            # Create new
            config = IntegrationConfig(
                type=integration_type,
                enabled=data.get("enabled", True),
                settings=settings,
                user_id=user_id,
                **notify_settings,
            )

        await store.save(config)
        logger.info(f"Integration configured: {integration_type} for user {user_id}")
        return json_response({"integration": config.to_dict()}, status=201 if not existing else 200)

    async def update_integration(
        self,
        integration_type: str,
        data: Dict[str, Any],
        user_id: str = "default",
    ) -> HandlerResult:
        """Partial update for integration (e.g., enable/disable).

        Args:
            integration_type: Type of integration
            data: Fields to update
            user_id: User/workspace ID

        Returns:
            Updated configuration
        """
        if integration_type not in VALID_INTEGRATION_TYPES:
            return error_response(f"Invalid integration type: {integration_type}", status=400)

        store = get_integration_store()
        config = await store.get(integration_type, user_id)

        if not config:
            return error_response(f"Integration not configured: {integration_type}", status=404)

        # Update allowed fields
        if "enabled" in data:
            config.enabled = bool(data["enabled"])

        for notify_key in [
            "notify_on_consensus",
            "notify_on_debate_end",
            "notify_on_error",
            "notify_on_leaderboard",
        ]:
            if notify_key in data:
                setattr(config, notify_key, bool(data[notify_key]))

        config.updated_at = time.time()
        await store.save(config)

        logger.info(f"Integration updated: {integration_type} for user {user_id}")
        return json_response({"integration": config.to_dict()})

    async def delete_integration(
        self, integration_type: str, user_id: str = "default"
    ) -> HandlerResult:
        """Delete integration configuration.

        Args:
            integration_type: Type of integration
            user_id: User/workspace ID

        Returns:
            Success message
        """
        if integration_type not in VALID_INTEGRATION_TYPES:
            return error_response(f"Invalid integration type: {integration_type}", status=400)

        store = get_integration_store()
        deleted = await store.delete(integration_type, user_id)

        if not deleted:
            return error_response(f"Integration not configured: {integration_type}", status=404)

        logger.info(f"Integration deleted: {integration_type} for user {user_id}")
        return json_response({"message": f"Integration {integration_type} deleted"})

    async def test_integration(
        self, integration_type: str, user_id: str = "default"
    ) -> HandlerResult:
        """Test integration connection.

        Args:
            integration_type: Type of integration
            user_id: User/workspace ID

        Returns:
            Test result
        """
        if integration_type not in VALID_INTEGRATION_TYPES:
            return error_response(f"Invalid integration type: {integration_type}", status=400)

        store = get_integration_store()
        config = await store.get(integration_type, user_id)

        if not config:
            return error_response(f"Integration not configured: {integration_type}", status=404)

        try:
            # Test based on integration type
            success = await self._test_connection(integration_type, config.settings)

            if success:
                config.last_activity = time.time()
                await store.save(config)
                return json_response(
                    {
                        "success": True,
                        "message": f"{integration_type} connection successful",
                    }
                )
            else:
                config.errors_24h += 1
                config.last_error = "Connection test failed"
                await store.save(config)
                return json_response(
                    {
                        "success": False,
                        "error": "Connection test failed",
                    }
                )

        except Exception as e:
            config.errors_24h += 1
            config.last_error = str(e)
            await store.save(config)
            logger.error(f"Integration test failed for {integration_type}: {e}")
            return json_response(
                {
                    "success": False,
                    "error": str(e),
                }
            )

    async def _test_connection(self, integration_type: str, settings: Dict[str, Any]) -> bool:
        """Test connection to integration provider.

        Args:
            integration_type: Type of integration
            settings: Provider settings

        Returns:
            True if connection successful
        """
        # Import integrations dynamically to avoid circular imports
        try:
            if integration_type == "slack":
                from aragora.integrations.slack import SlackConfig, SlackIntegration

                integration = SlackIntegration(
                    SlackConfig(  # type: ignore[call-arg]
                        webhook_url=settings.get("webhook_url", ""),
                    )
                )
                return await integration.verify_webhook()  # type: ignore[return-value,attr-defined]

            elif integration_type == "discord":
                from aragora.integrations.discord import DiscordConfig, DiscordIntegration

                integration = DiscordIntegration(  # type: ignore[assignment]
                    DiscordConfig(  # type: ignore[call-arg]
                        webhook_url=settings.get("webhook_url", ""),
                    )
                )
                return await integration.verify_webhook()  # type: ignore[return-value,attr-defined]

            elif integration_type == "telegram":
                from aragora.integrations.telegram import TelegramConfig, TelegramIntegration

                integration = TelegramIntegration(  # type: ignore[assignment]
                    TelegramConfig(  # type: ignore[call-arg]
                        bot_token=settings.get("bot_token", ""),
                        chat_id=settings.get("chat_id", ""),
                    )
                )
                return await integration.verify_connection()  # type: ignore[return-value,attr-defined]

            elif integration_type == "email":
                # For email, we just validate config structure
                provider = settings.get("provider", "smtp")
                if provider == "sendgrid" and not settings.get("sendgrid_api_key"):
                    return False
                if provider == "smtp" and not settings.get("smtp_host"):
                    return False
                return True

            elif integration_type == "teams":
                from aragora.integrations.teams import TeamsConfig, TeamsIntegration

                integration = TeamsIntegration(  # type: ignore[assignment]
                    TeamsConfig(  # type: ignore[call-arg]
                        webhook_url=settings.get("webhook_url", ""),
                    )
                )
                return await integration.verify_webhook()  # type: ignore[return-value,attr-defined]

            elif integration_type == "whatsapp":
                from aragora.integrations.whatsapp import WhatsAppConfig, WhatsAppIntegration

                integration = WhatsAppIntegration(  # type: ignore[assignment]
                    WhatsAppConfig(  # type: ignore[call-arg]
                        phone_number_id=settings.get("phone_number_id", ""),
                        access_token=settings.get("access_token", ""),
                        recipient=settings.get("recipient", ""),
                    )
                )
                return integration.is_configured  # type: ignore[attr-defined]

            elif integration_type == "matrix":
                from aragora.integrations.matrix import MatrixConfig, MatrixIntegration

                integration = MatrixIntegration(  # type: ignore[arg-type, assignment]
                    MatrixConfig(  # type: ignore[call-arg]
                        homeserver_url=settings.get("homeserver_url", ""),
                        access_token=settings.get("access_token", ""),
                        room_id=settings.get("room_id", ""),
                    )
                )
                return await integration.verify_connection()  # type: ignore[return-value,attr-defined]

            return False

        except ImportError as e:
            logger.warning(f"Integration module not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return False


# =============================================================================
# Route Registration Helper
# =============================================================================


def register_integration_routes(app: Any, handler: IntegrationsHandler) -> None:
    """Register integration routes with the application.

    Args:
        app: Application instance (aiohttp.web.Application)
        handler: IntegrationsHandler instance
    """
    from aiohttp import web

    async def get_status(request: web.Request) -> web.Response:
        user_id = request.get("user_id", "default")  # type: ignore[union-attr]
        result = await handler.get_status(user_id)
        return web.Response(
            body=result.body, status=result.status_code, content_type=result.content_type
        )

    async def get_integration(request: web.Request) -> web.Response:
        integration_type = request.match_info["type"]
        user_id = request.get("user_id", "default")  # type: ignore[union-attr]
        result = await handler.get_integration(integration_type, user_id)
        return web.Response(
            body=result.body, status=result.status_code, content_type=result.content_type
        )

    async def configure_integration(request: web.Request) -> web.Response:
        integration_type = request.match_info["type"]
        user_id = request.get("user_id", "default")  # type: ignore[union-attr]
        data = await request.json()
        result = await handler.configure_integration(integration_type, data, user_id)
        return web.Response(
            body=result.body, status=result.status_code, content_type=result.content_type
        )

    async def update_integration(request: web.Request) -> web.Response:
        integration_type = request.match_info["type"]
        user_id = request.get("user_id", "default")  # type: ignore[union-attr]
        data = await request.json()
        result = await handler.update_integration(integration_type, data, user_id)
        return web.Response(
            body=result.body, status=result.status_code, content_type=result.content_type
        )

    async def delete_integration(request: web.Request) -> web.Response:
        integration_type = request.match_info["type"]
        user_id = request.get("user_id", "default")  # type: ignore[union-attr]
        result = await handler.delete_integration(integration_type, user_id)
        return web.Response(
            body=result.body, status=result.status_code, content_type=result.content_type
        )

    async def test_integration(request: web.Request) -> web.Response:
        integration_type = request.match_info["type"]
        user_id = request.get("user_id", "default")  # type: ignore[union-attr]
        result = await handler.test_integration(integration_type, user_id)
        return web.Response(
            body=result.body, status=result.status_code, content_type=result.content_type
        )

    # Register versioned routes (v1)
    app.router.add_get("/api/v1/integrations/status", get_status)
    app.router.add_get("/api/v1/integrations/{type}", get_integration)
    app.router.add_put("/api/v1/integrations/{type}", configure_integration)
    app.router.add_patch("/api/v1/integrations/{type}", update_integration)
    app.router.add_delete("/api/v1/integrations/{type}", delete_integration)
    app.router.add_post("/api/v1/integrations/{type}/test", test_integration)

    # Register non-versioned routes (for frontend compatibility)
    app.router.add_get("/api/integrations/status", get_status)
    app.router.add_get("/api/integrations/{type}", get_integration)
    app.router.add_put("/api/integrations/{type}", configure_integration)
    app.router.add_patch("/api/integrations/{type}", update_integration)
    app.router.add_delete("/api/integrations/{type}", delete_integration)
    app.router.add_post("/api/integrations/{type}/test", test_integration)


__all__ = [
    "IntegrationsHandler",
    "IntegrationConfig",
    "IntegrationStatus",
    "IntegrationType",
    "VALID_INTEGRATION_TYPES",
    "register_integration_routes",
    "get_integration_store",
]
