"""
Unified Chat Webhook Router.

Routes incoming webhooks from all chat platforms (Slack, Teams, Discord,
Google Chat) to the appropriate connector and handler.

Endpoints:
- POST /api/chat/{platform}/webhook - Platform-specific webhook
- POST /api/chat/webhook - Auto-detect platform from headers
- GET  /api/chat/status - Get status of all configured platforms
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from aragora.connectors.chat import (
    ChatPlatformConnector,
    WebhookEvent,
    get_connector,
    get_configured_platforms,
    get_registry,
)

logger = logging.getLogger(__name__)

# Import base handler utilities
try:
    from ..base import (
        BaseHandler,
        HandlerResult,
        auto_error_response,
        error_response,
        json_response,
    )
    from ..utils.rate_limit import rate_limit

    HANDLER_BASE_AVAILABLE = True
except ImportError:
    HANDLER_BASE_AVAILABLE = False
    logger.warning("Handler base not available - ChatRouter will have limited functionality")


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed: {exc}", exc_info=exc)


def create_tracked_task(coro, name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


class ChatWebhookRouter:
    """
    Routes chat platform webhooks to appropriate handlers.

    Supports:
    - Platform-specific webhook endpoints
    - Auto-detection of platform from headers
    - Webhook signature verification
    - Event parsing and dispatching
    """

    # Supported platforms and their header signatures for auto-detection
    PLATFORM_SIGNATURES = {
        "slack": ["X-Slack-Signature", "X-Slack-Request-Timestamp"],
        "discord": ["X-Signature-Ed25519", "X-Signature-Timestamp"],
        "teams": ["Authorization"],  # Bot Framework uses Bearer tokens
        "google_chat": ["Authorization"],  # Google uses Bearer tokens
    }

    def __init__(
        self,
        event_handler: Optional[callable] = None,
        debate_starter: Optional[callable] = None,
    ):
        """
        Initialize the webhook router.

        Args:
            event_handler: Async function to handle parsed events
            debate_starter: Async function to start debates from commands
        """
        self.event_handler = event_handler
        self.debate_starter = debate_starter
        self._connectors: Dict[str, ChatPlatformConnector] = {}

    def get_connector(self, platform: str) -> Optional[ChatPlatformConnector]:
        """Get or create connector for a platform."""
        if platform not in self._connectors:
            connector = get_connector(platform)
            if connector:
                self._connectors[platform] = connector
        return self._connectors.get(platform)

    def detect_platform(self, headers: Dict[str, str]) -> Optional[str]:
        """Auto-detect platform from request headers."""
        # Check for Slack
        if headers.get("X-Slack-Signature"):
            return "slack"

        # Check for Discord
        if headers.get("X-Signature-Ed25519"):
            return "discord"

        # Teams and Google Chat both use Authorization headers
        # Check content for clues
        auth = headers.get("Authorization", "")
        if "Bearer" in auth:
            # Could be Teams or Google - need to check body
            # Default to teams as it's more common
            return "teams"

        return None

    def verify_webhook(
        self,
        platform: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> bool:
        """Verify webhook signature for a platform."""
        connector = self.get_connector(platform)
        if connector is None:
            logger.warning(f"No connector for platform: {platform}")
            return False

        return connector.verify_webhook(headers, body)

    def parse_event(
        self,
        platform: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse webhook payload into a WebhookEvent."""
        connector = self.get_connector(platform)
        if connector is None:
            return WebhookEvent(
                platform=platform,
                event_type="error",
                raw_payload={"error": "No connector available"},
            )

        return connector.parse_webhook_event(headers, body)

    async def handle_webhook(
        self,
        platform: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> Dict[str, Any]:
        """
        Handle an incoming webhook.

        Args:
            platform: Platform identifier
            headers: HTTP headers
            body: Raw request body

        Returns:
            Response dict with success status and any required response data
        """
        # Verify signature
        if not self.verify_webhook(platform, headers, body):
            logger.warning(f"Webhook verification failed for {platform}")
            return {"error": "Invalid signature", "status": 401}

        # Parse event
        event = self.parse_event(platform, headers, body)

        # Handle URL verification challenges
        if event.is_verification:
            logger.info(f"Handling verification challenge for {platform}")
            return self._handle_verification(platform, event)

        # Process the event
        response = await self._process_event(event)

        return response

    def _handle_verification(
        self,
        platform: str,
        event: WebhookEvent,
    ) -> Dict[str, Any]:
        """Handle URL verification challenges."""
        if platform == "slack":
            return {"challenge": event.challenge}

        if platform == "discord":
            # Discord expects type 1 PONG response
            return {"type": 1}

        if platform == "google_chat":
            # Google Chat verification is handled by auth
            return {"success": True}

        return {"success": True}

    async def _process_event(self, event: WebhookEvent) -> Dict[str, Any]:
        """Process a parsed webhook event."""
        logger.info(f"Processing {event.platform} event: {event.event_type}")

        # Handle commands (slash commands, bot mentions)
        if event.command:
            return await self._handle_command(event)

        # Handle interactions (button clicks, etc.)
        if event.interaction:
            return await self._handle_interaction(event)

        # Handle messages
        if event.message:
            return await self._handle_message(event)

        # Handle voice messages
        if event.voice_message:
            return await self._handle_voice(event)

        # Pass to generic event handler if configured
        if self.event_handler:
            try:
                await self.event_handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        return {"success": True}

    async def _handle_command(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle a bot command."""
        command = event.command
        if command is None:
            return {"success": True}

        logger.info(f"Command from {event.platform}: /{command.name} {command.args}")

        # Check for aragora-specific commands
        if command.name in ("aragora", "debate", "review", "gauntlet"):
            return await self._handle_aragora_command(event)

        # Pass to event handler
        if self.event_handler:
            try:
                await self.event_handler(event)
            except Exception as e:
                logger.error(f"Command handler error: {e}")

        return {"success": True}

    async def _handle_aragora_command(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle Aragora-specific commands."""
        command = event.command
        connector = self.get_connector(event.platform)

        if command is None or connector is None:
            return {"success": False, "error": "Invalid command context"}

        # Parse subcommand
        subcommand = command.args[0] if command.args else "help"
        args = command.args[1:] if len(command.args) > 1 else []

        response_text = ""
        blocks = None

        if subcommand == "help":
            response_text = self._get_help_text()
            blocks = connector.format_blocks(
                title="Aragora Commands",
                body=response_text,
            )

        elif subcommand == "status":
            status = await self._get_status()
            response_text = f"Aragora is {'online' if status['connected'] else 'offline'}"
            blocks = connector.format_blocks(
                title="Aragora Status",
                fields=[
                    ("Status", "Online" if status["connected"] else "Offline"),
                    ("Platforms", ", ".join(get_configured_platforms())),
                ],
            )

        elif subcommand in ("debate", "start"):
            # Start a debate
            topic = " ".join(args) if args else None
            if not topic:
                response_text = "Please provide a debate topic. Usage: /aragora debate <topic>"
            elif self.debate_starter:
                try:
                    result = await self.debate_starter(
                        topic=topic,
                        platform=event.platform,
                        channel=command.channel,
                        user=command.user,
                    )
                    response_text = f"Starting debate on: {topic}\nDebate ID: {result.get('debate_id', 'pending')}"
                except Exception as e:
                    response_text = f"Failed to start debate: {e}"
            else:
                response_text = "Debate starting not configured"

        else:
            response_text = f"Unknown command: {subcommand}\nUse /aragora help for available commands"

        # Send response
        try:
            await connector.respond_to_command(
                command,
                response_text,
                blocks=blocks,
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Failed to respond to command: {e}")

        return {"success": True}

    async def _handle_interaction(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle a user interaction."""
        interaction = event.interaction
        if interaction is None:
            return {"success": True}

        logger.info(f"Interaction from {event.platform}: {interaction.action_id}")

        # Pass to event handler
        if self.event_handler:
            try:
                await self.event_handler(event)
            except Exception as e:
                logger.error(f"Interaction handler error: {e}")

        return {"success": True}

    async def _handle_message(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle a regular message."""
        message = event.message
        if message is None:
            return {"success": True}

        # Skip bot messages
        if message.author.is_bot:
            return {"success": True}

        logger.debug(f"Message from {event.platform}: {message.content[:50]}...")

        # Pass to event handler
        if self.event_handler:
            try:
                await self.event_handler(event)
            except Exception as e:
                logger.error(f"Message handler error: {e}")

        return {"success": True}

    async def _handle_voice(self, event: WebhookEvent) -> Dict[str, Any]:
        """Handle a voice message."""
        voice = event.voice_message
        if voice is None:
            return {"success": True}

        logger.info(f"Voice message from {event.platform}: {voice.duration_seconds}s")

        # Transcribe using voice bridge
        try:
            from aragora.connectors.chat import get_voice_bridge

            bridge = get_voice_bridge()
            connector = self.get_connector(event.platform)

            if connector:
                transcription = await bridge.transcribe_voice_message(
                    voice,
                    connector=connector,
                )
                logger.info(f"Transcribed: {transcription[:100]}...")

                # Create a message event with transcription
                event.message = event.voice_message = None
                # Could trigger debate or pass to handler
        except Exception as e:
            logger.error(f"Voice transcription error: {e}")

        return {"success": True}

    def _get_help_text(self) -> str:
        """Get help text for Aragora commands."""
        return """
*Available Commands:*
- `/aragora help` - Show this help message
- `/aragora status` - Check Aragora status
- `/aragora debate <topic>` - Start a multi-agent debate
- `/aragora review <url>` - Request a code review
- `/aragora gauntlet <topic>` - Run a stress test

*Examples:*
- `/aragora debate Should we use microservices?`
- `/aragora review https://github.com/org/repo/pull/123`
"""

    async def _get_status(self) -> Dict[str, Any]:
        """Get Aragora service status."""
        platforms = get_configured_platforms()
        return {
            "connected": len(platforms) > 0,
            "platforms": platforms,
            "timestamp": __import__("time").time(),
        }


# Handler class for integration with server framework
if HANDLER_BASE_AVAILABLE:

    class ChatHandler(BaseHandler):
        """HTTP handler for chat webhooks."""

        ROUTES = [
            "/api/chat/webhook",
            "/api/chat/status",
            "/api/chat/slack/webhook",
            "/api/chat/teams/webhook",
            "/api/chat/discord/webhook",
            "/api/chat/google_chat/webhook",
        ]

        def __init__(self):
            """Initialize with router."""
            self.router = ChatWebhookRouter()

        def can_handle(self, path: str, method: str = "GET") -> bool:
            """Check if this handler can process the given path."""
            return path in self.ROUTES or path.startswith("/api/chat/")

        def handle(
            self, path: str, query_params: Dict[str, Any], handler: Any
        ) -> Optional[HandlerResult]:
            """Route chat requests."""
            logger.debug(f"Chat request: {path}")

            if path == "/api/chat/status":
                return self._get_status()

            # All webhook endpoints require POST
            if handler.command != "POST":
                return error_response("Method not allowed", 405)

            return None  # Let handle_post handle it

        def handle_post(
            self, path: str, body: Dict[str, Any], handler: Any
        ) -> Optional[HandlerResult]:
            """Handle POST requests (webhooks)."""
            # Get headers
            headers = {k: v for k, v in handler.headers.items()}

            # Get raw body
            content_length = int(headers.get("Content-Length", 0))
            if hasattr(handler, "rfile"):
                raw_body = handler.rfile.read(content_length)
            else:
                raw_body = json.dumps(body).encode()

            # Determine platform
            if "/slack/" in path:
                platform = "slack"
            elif "/teams/" in path:
                platform = "teams"
            elif "/discord/" in path:
                platform = "discord"
            elif "/google_chat/" in path:
                platform = "google_chat"
            else:
                platform = self.router.detect_platform(headers)

            if not platform:
                return error_response("Could not determine platform", 400)

            # Handle webhook asynchronously
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self.router.handle_webhook(platform, headers, raw_body)
            )

            if "error" in result:
                return error_response(result["error"], result.get("status", 400))

            return json_response(result)

        def _get_status(self) -> HandlerResult:
            """Get chat integration status."""
            platforms = get_configured_platforms()
            registry = get_registry()

            status = {
                "enabled": len(platforms) > 0,
                "configured_platforms": platforms,
                "connectors": {
                    name: {
                        "name": conn.platform_display_name,
                        "configured": conn.is_configured(),
                    }
                    for name, conn in registry.all().items()
                },
            }

            return json_response(status)


# Singleton router instance
_router: Optional[ChatWebhookRouter] = None


def get_webhook_router(
    event_handler: Optional[callable] = None,
    debate_starter: Optional[callable] = None,
) -> ChatWebhookRouter:
    """Get or create the webhook router singleton."""
    global _router
    if _router is None:
        _router = ChatWebhookRouter(event_handler, debate_starter)
    return _router
