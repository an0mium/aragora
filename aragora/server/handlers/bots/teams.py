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
import logging
import os
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    HandlerResult,
    json_response,
)
from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.secure import SecureHandler
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Environment variables
TEAMS_APP_ID = os.environ.get("TEAMS_APP_ID", "")
TEAMS_APP_PASSWORD = os.environ.get("TEAMS_APP_PASSWORD", "")


def _check_botframework_available() -> tuple[bool, Optional[str]]:
    """Check if Bot Framework SDK is available."""
    try:
        from botbuilder.core import TurnContext  # noqa: F401 - availability check

        return True, None
    except ImportError:
        return False, "botbuilder-core not installed"


class TeamsHandler(BotHandlerMixin, SecureHandler):
    """Handler for Microsoft Teams Bot endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Message webhook endpoints are authenticated via Bot Framework,
    not RBAC, since they are called by Microsoft servers directly.
    """

    # BotHandlerMixin configuration
    bot_platform = "teams"

    ROUTES = [
        "/api/v1/bots/teams/messages",
        "/api/v1/bots/teams/status",
    ]

    def __init__(self, ctx: dict = None):  # type: ignore[assignment]
        super().__init__(ctx or {})  # type: ignore[arg-type]
        self._bot: Optional[Any] = None
        self._bot_initialized = False

    def _is_bot_enabled(self) -> bool:
        """Check if Teams bot is configured."""
        return bool(TEAMS_APP_ID and TEAMS_APP_PASSWORD)

    def _build_status_response(
        self, extra_status: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """Build Teams-specific status response."""
        available, error = _check_botframework_available()

        status = {
            "platform": self.bot_platform,
            "enabled": self._is_bot_enabled(),
            "app_id_configured": bool(TEAMS_APP_ID),
            "password_configured": bool(TEAMS_APP_PASSWORD),
            "sdk_available": available,
            "sdk_error": error,
        }
        if extra_status:
            status.update(extra_status)
        return json_response(status)

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
        except ImportError as e:
            logger.warning(f"Teams bot module not available: {e}")
            self._bot = None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to initialize Teams bot due to configuration error: {e}")
            self._bot = None
        except Exception as e:
            logger.exception(f"Unexpected error initializing Teams bot: {e}")
            self._bot = None

        return self._bot

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=30, limiter_name="teams_status")
    async def handle(  # type: ignore[override]
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Teams requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/teams/status":
            # Use BotHandlerMixin's RBAC-protected status handler
            return await self.handle_status_request(handler)

        return None

    @rate_limit(rpm=60, limiter_name="teams_messages")
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests."""
        if path == "/api/v1/bots/teams/messages":
            return self._handle_messages(handler)

        return None

    def _handle_messages(self, handler: Any) -> HandlerResult:
        """Handle incoming Bot Framework messages.

        This endpoint receives activities from Microsoft Teams via the
        Bot Framework Service.
        """
        bot = self._ensure_bot()
        if not bot:
            return json_response(
                {
                    "error": "Teams bot not configured",
                    "details": "Set TEAMS_APP_ID and TEAMS_APP_PASSWORD environment variables",
                },
                status=503,
            )

        try:
            # Read and parse body
            body = self._read_request_body(handler)
            activity_json, err = self._parse_json_body(body, "Teams message")
            if err:
                return err

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
                    await adapter.authenticate_request(activity, auth_header)
                except (ValueError, KeyError) as auth_error:
                    logger.warning(f"Teams auth failed due to invalid token: {auth_error}")
                    self._audit_webhook_auth_failure("auth_token", "invalid_token")
                    response_status = 401
                    response_body = {"error": "Invalid authentication token"}
                    return
                except Exception as auth_error:
                    logger.exception(f"Unexpected Teams auth error: {auth_error}")
                    self._audit_webhook_auth_failure("auth_token", "unexpected_error")
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
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Data error processing Teams activity: {e}")
                    response_status = 400
                    response_body = {"error": str(e)[:100]}
                except Exception as e:
                    logger.exception(f"Unexpected error processing Teams activity: {e}")
                    response_status = 500
                    response_body = {"error": "Internal processing error"}

            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(process_activity())
            finally:
                loop.close()

            return json_response(response_body, status=response_status)

        except Exception as e:
            return self._handle_webhook_exception(e, "Teams message", return_200_on_error=False)


__all__ = ["TeamsHandler"]
