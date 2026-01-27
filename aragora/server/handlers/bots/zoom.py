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

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from aragora.audit.unified import audit_security
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# RBAC permission for bot configuration endpoints
BOTS_READ_PERMISSION = "bots:read"

# Environment variables
ZOOM_CLIENT_ID = os.environ.get("ZOOM_CLIENT_ID", "")
ZOOM_CLIENT_SECRET = os.environ.get("ZOOM_CLIENT_SECRET", "")
ZOOM_BOT_JID = os.environ.get("ZOOM_BOT_JID", "")
ZOOM_SECRET_TOKEN = os.environ.get("ZOOM_SECRET_TOKEN", "")


class ZoomHandler(SecureHandler):
    """Handler for Zoom Bot endpoints.

    RBAC Protected:
    - bots:read - required for status endpoint

    Note: Event webhook endpoints are authenticated via Zoom's signature,
    not RBAC, since they are called by Zoom servers directly.
    """

    ROUTES = [
        "/api/v1/bots/zoom/events",
        "/api/v1/bots/zoom/status",
    ]

    def __init__(self, ctx: dict = None):  # type: ignore[assignment]
        super().__init__(ctx or {})  # type: ignore[arg-type]
        self._bot: Optional[Any] = None
        self._bot_initialized = False

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
    async def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Zoom requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/zoom/status":
            # RBAC: Require authentication and bots:read permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, BOTS_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning(f"Zoom status access denied: {e}")
                return error_response(str(e), 403)
            return self._get_status()

        return None

    @rate_limit(rpm=30)
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/bots/zoom/events":
            return self._handle_events(handler)

        return None

    def _get_status(self) -> HandlerResult:
        """Get Zoom bot status."""
        return json_response(
            {
                "enabled": bool(ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET),
                "client_id_configured": bool(ZOOM_CLIENT_ID),
                "client_secret_configured": bool(ZOOM_CLIENT_SECRET),
                "bot_jid_configured": bool(ZOOM_BOT_JID),
                "secret_token_configured": bool(ZOOM_SECRET_TOKEN),
            }
        )

    def _handle_events(self, handler: Any) -> HandlerResult:
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
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            # Verify signature if bot is configured
            if bot and signature and not bot.verify_webhook(body, timestamp, signature):
                logger.warning("Zoom webhook signature verification failed")
                audit_security(
                    event_type="zoom_webhook_auth_failed",
                    actor_id="unknown",
                    resource_type="zoom_webhook",
                    resource_id="signature",
                )
                return error_response("Invalid signature", 401)

            # Parse event
            try:
                event = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Zoom event: {e}")
                return error_response("Invalid JSON", 400)

            event_type = event.get("event", "")
            logger.info(f"Zoom event received: {event_type}")

            # Handle URL validation even without full bot
            if event_type == "endpoint.url_validation":
                payload = event.get("payload", {})
                plain_token = payload.get("plainToken", "")

                if ZOOM_SECRET_TOKEN:
                    import hashlib
                    import hmac

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
                else:
                    return json_response({"plainToken": plain_token})

            # For other events, require bot
            if not bot:
                return json_response(
                    {
                        "error": "Zoom bot not configured",
                        "details": "Set ZOOM_CLIENT_ID and ZOOM_CLIENT_SECRET environment variables",
                    },
                    status=503,
                )

            # Process event asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(bot.handle_event(event))
            finally:
                loop.close()

            return json_response(result)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Zoom event: {e}")
            return error_response("Invalid JSON payload", 400)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Data error in Zoom event: {e}")
            return error_response(safe_error_message(e, "Zoom event"), 400)
        except (ConnectionError, OSError, TimeoutError) as e:
            logger.error(f"Connection error processing Zoom event: {e}")
            return error_response(safe_error_message(e, "Zoom event"), 503)
        except Exception as e:
            logger.exception(f"Unexpected Zoom event error: {e}")
            return error_response(safe_error_message(e, "Zoom event"), 500)


__all__ = ["ZoomHandler"]
