"""
Zoom Bot endpoint handler.

Handles incoming events from Zoom's webhook API for chat messages
and meeting events.

Endpoints:
- POST /api/bots/zoom/events - Handle Zoom events/webhooks
- GET /api/bots/zoom/status - Bot status

Environment Variables:
- ZOOM_CLIENT_ID - Required for OAuth
- ZOOM_CLIENT_SECRET - Required for OAuth
- ZOOM_BOT_JID - Bot's JID
- ZOOM_SECRET_TOKEN - Webhook signature verification
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Environment variables - None defaults make misconfiguration explicit
ZOOM_CLIENT_ID = os.environ.get("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.environ.get("ZOOM_CLIENT_SECRET")
ZOOM_BOT_JID = os.environ.get("ZOOM_BOT_JID")
ZOOM_SECRET_TOKEN = os.environ.get("ZOOM_SECRET_TOKEN")

# Log warnings at module load time for missing secrets
if not ZOOM_SECRET_TOKEN:
    logger.warning("ZOOM_SECRET_TOKEN not configured - webhook signature verification disabled")


class ZoomHandler(BotHandlerMixin, SecureHandler):
    """Handler for Zoom Bot endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Event webhook endpoints are authenticated via Zoom's signature,
    not RBAC, since they are called by Zoom servers directly.
    """

    # BotHandlerMixin configuration
    bot_platform = "zoom"

    ROUTES = [
        "/api/v1/bots/zoom/events",
        "/api/v1/bots/zoom/status",
    ]

    def __init__(self, ctx: dict = None):  # type: ignore[assignment]
        super().__init__(ctx or {})  # type: ignore[arg-type]
        self._bot: Optional[Any] = None
        self._bot_initialized = False

    def _is_bot_enabled(self) -> bool:
        """Check if Zoom bot is configured."""
        return bool(ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET)

    def _build_status_response(
        self, extra_status: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """Build Zoom-specific status response."""
        status = {
            "platform": self.bot_platform,
            "enabled": self._is_bot_enabled(),
            "client_id_configured": bool(ZOOM_CLIENT_ID),
            "client_secret_configured": bool(ZOOM_CLIENT_SECRET),
            "bot_jid_configured": bool(ZOOM_BOT_JID),
            "secret_token_configured": bool(ZOOM_SECRET_TOKEN),
        }
        if extra_status:
            status.update(extra_status)
        return json_response(status)

    def _ensure_bot(self) -> Optional[Any]:
        """Lazily initialize the Zoom bot."""
        if self._bot_initialized:
            return self._bot

        self._bot_initialized = True

        if not ZOOM_CLIENT_ID or not ZOOM_CLIENT_SECRET:
            logger.warning("Zoom credentials not configured")
            return None

        try:
            from aragora.bots.zoom_bot import create_zoom_bot

            self._bot = create_zoom_bot()
            logger.info("Zoom bot initialized")
        except ImportError as e:
            logger.warning(f"Zoom bot module not available: {e}")
            self._bot = None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to initialize Zoom bot due to configuration error: {e}")
            self._bot = None
        except Exception as e:
            logger.exception(f"Unexpected error initializing Zoom bot: {e}")
            self._bot = None

        return self._bot

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=30)
    async def handle(  # type: ignore[override]
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Zoom requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/zoom/status":
            # Use BotHandlerMixin's RBAC-protected status handler
            return await self.handle_status_request(handler)

        return None

    @rate_limit(rpm=30)
    async def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/bots/zoom/events":
            return await self._handle_events(handler)

        return None

    async def _handle_events(self, handler: Any) -> HandlerResult:
        """Handle incoming Zoom webhook events.

        This endpoint receives events from Zoom including:
        - endpoint.url_validation (initial verification)
        - bot_notification (chat messages)
        - meeting.ended (meeting ended)
        - bot_installed (bot was installed)
        """
        bot = self._ensure_bot()
        if not bot:
            # For URL validation, we still need to respond even without full bot
            pass

        try:
            # Get verification headers
            timestamp = handler.headers.get("x-zm-request-timestamp", "")
            signature = handler.headers.get("x-zm-signature", "")

            # Read body
            body = self._read_request_body(handler)

            # Parse event first to check if URL validation (which has different requirements)
            event, err = self._parse_json_body(body, "Zoom event")
            if err:
                return err

            event_type = event.get("event", "")
            logger.info(f"Zoom event received: {event_type}")

            # Handle URL validation - requires ZOOM_SECRET_TOKEN
            if event_type == "endpoint.url_validation":
                if not ZOOM_SECRET_TOKEN:
                    logger.warning("ZOOM_SECRET_TOKEN not configured - rejecting URL validation")
                    return error_response("Zoom secret token not configured", 503)

                import hashlib
                import hmac

                payload = event.get("payload", {})
                plain_token = payload.get("plainToken", "")

                encrypted = hmac.new(
                    ZOOM_SECRET_TOKEN.encode(),
                    plain_token.encode(),
                    hashlib.sha256,
                ).hexdigest()
                return json_response(
                    {
                        "plainToken": plain_token,
                        "encryptedToken": encrypted,
                    }
                )

            # For all other events, require signature verification
            if signature:
                if not bot:
                    logger.warning("Zoom bot not configured - cannot verify signature")
                    return error_response("Zoom bot not configured for signature verification", 503)
                if not bot.verify_webhook(body, timestamp, signature):
                    logger.warning("Zoom webhook signature verification failed")
                    self._audit_webhook_auth_failure("signature")
                    return error_response("Invalid signature", 401)
            else:
                # No signature provided - for security, require it
                logger.warning("Zoom webhook request missing signature")
                self._audit_webhook_auth_failure("signature", "missing")
                return error_response("Missing signature header", 401)

            # For other events, require bot
            if not bot:
                return json_response(
                    {
                        "error": "Zoom bot not configured",
                        "details": "Set ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET environment variables",
                    },
                    status=503,
                )

            # Process event
            result = await bot.handle_event(event)
            return json_response(result)

        except Exception as e:
            return self._handle_webhook_exception(e, "Zoom event", return_200_on_error=False)


__all__ = ["ZoomHandler"]
