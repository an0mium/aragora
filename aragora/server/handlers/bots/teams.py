"""
Microsoft Teams Bot endpoint handler.

Handles incoming messages and card actions from Microsoft Teams
via the Bot Framework messaging endpoint.

Endpoints:
- POST /api/bots/teams/messages - Handle incoming messages
- GET /api/bots/teams/status - Bot status

Environment Variables:
- TEAMS_APP_ID - Required for bot authentication
- TEAMS_APP_PASSWORD - Required for bot authentication
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)

logger = logging.getLogger(__name__)

# Environment variables
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID", "")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD", "")


def _check_botframework_available() -> tuple[bool, Optional[str]]:
    """Check if Bot Framework SDK is available."""
    try:
        from botbuilder.core import TurnContext
        return True, None
    except ImportError:
        return False, "botbuilder-core not installed"


class TeamsHandler(BaseHandler):
    """Handler for Microsoft Teams Bot endpoints."""

    ROUTES = [
        "/api/bots/teams/messages",
        "/api/bots/teams/status",
    ]

    def __init__(self):
        self._bot: Optional[Any] = None
        self._bot_initialized = False

    def _ensure_bot(self) -> Optional[Any]:
        """Lazily initialize the Teams bot."""
        if self._bot_initialized:
            return self._bot

        self._bot_initialized = True

        if not TEAMS_APP_ID or not TEAMS_APP_PASSWORD:
            logger.warning("Teams credentials not configured")
            return None

        available, _ = _check_botframework_available()
        if not available:
            logger.warning("Bot Framework SDK not available")
            return None

        try:
            from aragora.bots.teams_bot import create_teams_bot
            self._bot = create_teams_bot()

            # Run setup synchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._bot.setup())
            finally:
                loop.close()

            logger.info("Teams bot initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Teams bot: {e}")
            self._bot = None

        return self._bot

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Teams requests."""
        if path == "/api/bots/teams/status":
            return self._get_status()

        return None

    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/bots/teams/messages":
            return self._handle_messages(handler)

        return None

    def _get_status(self) -> HandlerResult:
        """Get Teams bot status."""
        available, error = _check_botframework_available()

        return json_response({
            "enabled": bool(TEAMS_APP_ID and TEAMS_APP_PASSWORD),
            "app_id_configured": bool(TEAMS_APP_ID),
            "password_configured": bool(TEAMS_APP_PASSWORD),
            "sdk_available": available,
            "sdk_error": error,
        })

    def _handle_messages(self, handler: Any) -> HandlerResult:
        """Handle incoming Bot Framework messages.

        This endpoint receives activities from Microsoft Teams via the
        Bot Framework Service.
        """
        bot = self._ensure_bot()
        if not bot:
            return json_response({
                "error": "Teams bot not configured",
                "details": "Set TEAMS_APP_ID and TEAMS_APP_PASSWORD environment variables",
            }, status_code=503)

        try:
            # Read body
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            # Parse activity
            try:
                activity_json = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Teams message: {e}")
                return error_response("Invalid JSON", 400)

            # Get authorization header
            auth_header = handler.headers.get("Authorization", "")

            # Process with Bot Framework adapter
            from botbuilder.schema import Activity

            activity = Activity.deserialize(activity_json)

            # Create response
            response_body: Dict[str, Any] = {}
            response_status = 200

            async def process_activity():
                nonlocal response_body, response_status

                adapter = bot.get_adapter()

                # Authenticate the request
                try:
                    claims_identity = await adapter.authenticate_request(
                        activity, auth_header
                    )
                except Exception as auth_error:
                    logger.warning(f"Teams auth failed: {auth_error}")
                    response_status = 401
                    response_body = {"error": "Unauthorized"}
                    return

                # Process the activity
                try:
                    await adapter.process_activity(
                        activity,
                        auth_header,
                        bot.on_turn,
                    )
                except Exception as e:
                    logger.error(f"Error processing Teams activity: {e}", exc_info=True)
                    response_status = 500
                    response_body = {"error": str(e)[:100]}

            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(process_activity())
            finally:
                loop.close()

            return json_response(response_body, status=response_status)

        except Exception as e:
            logger.error(f"Teams message error: {e}", exc_info=True)
            return error_response(safe_error_message(e, "teams message"), 500)


__all__ = ["TeamsHandler"]
