"""
Unified Chat Webhook Router.

Routes incoming webhooks from all chat platforms (Slack, Teams, Discord,
Google Chat, Telegram, WhatsApp) to the appropriate connector and handler.

Endpoints:
- POST /api/chat/{platform}/webhook - Platform-specific webhook
- POST /api/chat/webhook - Auto-detect platform from headers
- GET  /api/chat/status - Get status of all configured platforms

Supported platforms:
- Slack: /api/chat/slack/webhook
- Teams: /api/chat/teams/webhook
- Discord: /api/chat/discord/webhook
- Google Chat: /api/chat/google_chat/webhook
- Telegram: /api/chat/telegram/webhook
- WhatsApp: /api/chat/whatsapp/webhook
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional

from aragora.connectors.chat import (
    ChatPlatformConnector,
    WebhookEvent,
    get_connector,
    get_configured_platforms,
    get_registry,
)

# DecisionRouter for unified routing
try:
    from aragora.core.decision import (
        DecisionRequest,
        DecisionRouter,
        DecisionType,
        InputSource,
        ResponseChannel,
        RequestContext,
        get_decision_router,
    )

    DECISION_ROUTER_AVAILABLE = True
except ImportError:
    DECISION_ROUTER_AVAILABLE = False
    DecisionRequest = None  # type: ignore[misc]
    DecisionRouter = None  # type: ignore[misc]
    DecisionType = None  # type: ignore[misc]
    InputSource = None  # type: ignore[misc]
    ResponseChannel = None  # type: ignore[misc]
    RequestContext = None  # type: ignore[misc]
    get_decision_router = None

logger = logging.getLogger(__name__)

# Import base handler utilities
try:
    from ..base import (
        BaseHandler,
        HandlerResult,
        auto_error_response,  # noqa: F401
        error_response,
        json_response,
    )
    from ..utils.rate_limit import RateLimiter, get_client_ip, rate_limit  # noqa: F401

    HANDLER_BASE_AVAILABLE = True
    # Rate limiter for chat webhook endpoints (60 requests per minute)
    _chat_limiter = RateLimiter(requests_per_minute=60)
except ImportError:
    HANDLER_BASE_AVAILABLE = False
    _chat_limiter = None
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
        "telegram": ["X-Telegram-Bot-Api-Secret-Token"],  # Telegram webhook secret
        "whatsapp": ["X-Hub-Signature-256"],  # WhatsApp/Meta signature
    }

    def __init__(
        self,
        event_handler: Optional[Callable[..., Any]] = None,
        debate_starter: Optional[Callable[..., Any]] = None,
        decision_router: Optional[Any] = None,
    ):
        """
        Initialize the webhook router.

        Args:
            event_handler: Async function to handle parsed events
            debate_starter: Async function to start debates from commands
            decision_router: Optional DecisionRouter for unified routing
        """
        self.event_handler = event_handler
        self.debate_starter = debate_starter
        self._connectors: Dict[str, ChatPlatformConnector] = {}

        # Initialize DecisionRouter if available
        self._decision_router = decision_router
        if self._decision_router is None and DECISION_ROUTER_AVAILABLE:
            try:
                self._decision_router = get_decision_router()
            except Exception as e:
                logger.debug(f"DecisionRouter not available: {e}")

    def get_connector(self, platform: str) -> Optional[ChatPlatformConnector]:
        """Get or create connector for a platform."""
        if platform not in self._connectors:
            connector = get_connector(platform)
            if connector:
                self._connectors[platform] = connector
        return self._connectors.get(platform)

    def detect_platform(self, headers: Dict[str, str]) -> Optional[str]:
        """Auto-detect platform from request headers."""
        # Check for Slack (most reliable - unique signature header)
        if headers.get("X-Slack-Signature"):
            return "slack"

        # Check for Discord (unique signature header)
        if headers.get("X-Signature-Ed25519"):
            return "discord"

        # Check for Telegram secret token (if configured)
        if headers.get("X-Telegram-Bot-Api-Secret-Token"):
            return "telegram"

        # Check for WhatsApp/Meta signature
        # Note: X-Hub-Signature-256 is used by Meta (Facebook/Instagram/WhatsApp)
        if headers.get("X-Hub-Signature-256"):
            return "whatsapp"

        # Teams and Google Chat both use Authorization headers
        auth = headers.get("Authorization", "")
        if "Bearer" in auth:
            # Could be Teams or Google - need to check body later
            return "teams"

        return None

    def detect_platform_from_body(
        self,
        headers: Dict[str, str],
        body: bytes,
    ) -> Optional[str]:
        """
        Auto-detect platform from request body structure.

        Used as fallback when headers don't provide definitive identification.
        This is particularly useful for:
        - Telegram webhooks without secret token configured
        - Distinguishing WhatsApp from other Meta webhooks
        - Distinguishing Teams from Google Chat
        """
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        # Telegram detection: has update_id and optional message/callback_query
        # Example: {"update_id": 123, "message": {...}} or {"update_id": 123, "callback_query": {...}}
        if "update_id" in payload:
            if any(
                k in payload
                for k in (
                    "message",
                    "callback_query",
                    "inline_query",
                    "edited_message",
                    "channel_post",
                )
            ):
                return "telegram"

        # WhatsApp/Meta webhook structure
        # Example: {"object": "whatsapp_business_account", "entry": [...]}
        if payload.get("object") == "whatsapp_business_account":
            return "whatsapp"

        # Facebook/Instagram webhooks also use X-Hub-Signature-256 but different object
        if payload.get("object") in ("page", "instagram"):
            # Not WhatsApp - could add separate handling if needed
            return None

        # Microsoft Teams Bot Framework
        # Example: {"type": "message", "serviceUrl": "https://smba.trafficmanager.net/...", ...}
        if payload.get("serviceUrl") and "trafficmanager.net" in payload.get("serviceUrl", ""):
            return "teams"
        if payload.get("channelId") == "msteams":
            return "teams"

        # Google Chat webhook structure
        # Example: {"type": "MESSAGE", "message": {...}, "space": {...}, "configCompleteRedirectUrl": ...}
        if payload.get("type") in (
            "MESSAGE",
            "ADDED_TO_SPACE",
            "REMOVED_FROM_SPACE",
            "CARD_CLICKED",
        ):
            if "space" in payload or "message" in payload:
                return "google_chat"

        # Discord interaction (not signature-based)
        # Example: {"type": 1, "application_id": "...", ...}
        if "application_id" in payload and "type" in payload:
            if isinstance(payload.get("type"), int):
                return "discord"

        # Slack (for non-signed requests like certain slash commands)
        # Example: {"token": "...", "team_id": "...", "api_app_id": "..."}
        if all(k in payload for k in ("token", "team_id", "api_app_id")):
            return "slack"

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

        if platform == "telegram":
            # Telegram webhook verification is done via setWebhook call
            return {"success": True}

        if platform == "whatsapp":
            # WhatsApp webhook verification requires hub.challenge echo
            if event.challenge:
                return {"hub.challenge": event.challenge}
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
            elif self._decision_router and DECISION_ROUTER_AVAILABLE:
                # Use DecisionRouter for unified routing
                try:
                    result = await self._route_decision(
                        content=topic,
                        decision_type=DecisionType.DEBATE,
                        event=event,
                        command=command,
                    )
                    response_text = f"Starting debate on: {topic}\nDebate ID: {result.decision_id}"
                except Exception as e:
                    logger.error(f"DecisionRouter error: {e}")
                    response_text = f"Failed to start debate: {e}"
            elif self.debate_starter:
                # Fallback to legacy debate_starter
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

        elif subcommand == "gauntlet":
            # Run gauntlet stress-test via DecisionRouter
            topic = " ".join(args) if args else None
            if not topic:
                response_text = "Please provide a topic. Usage: /aragora gauntlet <topic>"
            elif self._decision_router and DECISION_ROUTER_AVAILABLE:
                try:
                    result = await self._route_decision(
                        content=topic,
                        decision_type=DecisionType.GAUNTLET,
                        event=event,
                        command=command,
                    )
                    response_text = f"Running gauntlet on: {topic}\nID: {result.decision_id}"
                except Exception as e:
                    logger.error(f"DecisionRouter error: {e}")
                    response_text = f"Failed to run gauntlet: {e}"
            else:
                response_text = "Gauntlet not available (DecisionRouter required)"

        elif subcommand == "workflow":
            # Start workflow via DecisionRouter
            workflow_name = args[0] if args else None
            if not workflow_name:
                response_text = "Please provide a workflow name. Usage: /aragora workflow <name>"
            elif self._decision_router and DECISION_ROUTER_AVAILABLE:
                try:
                    result = await self._route_decision(
                        content=workflow_name,
                        decision_type=DecisionType.WORKFLOW,
                        event=event,
                        command=command,
                    )
                    response_text = f"Starting workflow: {workflow_name}\nID: {result.decision_id}"
                except Exception as e:
                    logger.error(f"DecisionRouter error: {e}")
                    response_text = f"Failed to start workflow: {e}"
            else:
                response_text = "Workflow not available (DecisionRouter required)"

        else:
            response_text = (
                f"Unknown command: {subcommand}\nUse /aragora help for available commands"
            )

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
- `/aragora gauntlet <topic>` - Run adversarial validation
- `/aragora workflow <name>` - Start a workflow
- `/aragora review <url>` - Request a code review

*Examples:*
- `/aragora debate Should we use microservices?`
- `/aragora gauntlet Is this contract compliant?`
- `/aragora workflow security-audit`
"""

    async def _get_status(self) -> Dict[str, Any]:
        """Get Aragora service status."""
        platforms = get_configured_platforms()
        return {
            "connected": len(platforms) > 0,
            "platforms": platforms,
            "timestamp": __import__("time").time(),
        }

    def _get_input_source(self, platform: str) -> Any:
        """Map platform string to InputSource enum."""
        if not DECISION_ROUTER_AVAILABLE or InputSource is None:
            return None

        platform_map = {
            "slack": InputSource.SLACK,
            "discord": InputSource.DISCORD,
            "teams": InputSource.TEAMS,
            "google_chat": InputSource.GOOGLE_CHAT,
            "telegram": InputSource.TELEGRAM,
            "whatsapp": InputSource.WHATSAPP,
        }
        return platform_map.get(platform.lower(), InputSource.HTTP_API)

    async def _route_decision(
        self,
        content: str,
        decision_type: Any,
        event: WebhookEvent,
        command: Any,
    ) -> Any:
        """Route a decision request through the DecisionRouter."""
        if not DECISION_ROUTER_AVAILABLE or self._decision_router is None:
            raise RuntimeError("DecisionRouter not available")

        # Build response channel
        response_channel = ResponseChannel(
            platform=event.platform,
            channel_id=command.channel if command else None,
            user_id=command.user.id if command and command.user else None,
            thread_id=getattr(command, "thread_ts", None) if command else None,
            response_format="full",
            include_reasoning=True,
        )

        # Build request context
        context = RequestContext(
            user_id=command.user.id if command and command.user else None,
            user_name=command.user.name if command and command.user else None,
            org_id=getattr(command, "team_id", None) if command else None,
            session_id=command.channel if command else None,
        )

        # Create decision request
        request = DecisionRequest(
            content=content,
            decision_type=decision_type,
            source=self._get_input_source(event.platform),
            response_channels=[response_channel],
            context=context,
        )

        # Route through DecisionRouter
        result = await self._decision_router.route(request)
        return result


# Handler class for integration with server framework
if HANDLER_BASE_AVAILABLE:

    class ChatHandler(BaseHandler):
        """HTTP handler for chat webhooks."""

        ROUTES = [
            "/api/v1/chat/webhook",
            "/api/v1/chat/status",
            "/api/v1/chat/slack/webhook",
            "/api/v1/chat/teams/webhook",
            "/api/v1/chat/discord/webhook",
            "/api/v1/chat/google_chat/webhook",
            "/api/v1/chat/telegram/webhook",
            "/api/v1/chat/whatsapp/webhook",
        ]

        def __init__(self):
            """Initialize with router."""
            self.router = ChatWebhookRouter()

        def can_handle(self, path: str, method: str = "GET") -> bool:
            """Check if this handler can process the given path."""
            return path in self.ROUTES or path.startswith("/api/v1/chat/")

        def handle(
            self, path: str, query_params: Dict[str, Any], handler: Any
        ) -> Optional[HandlerResult]:
            """Route chat requests."""
            logger.debug(f"Chat request: {path}")

            if path == "/api/v1/chat/status":
                return self._get_status()

            # All webhook endpoints require POST
            if handler.command != "POST":
                return error_response("Method not allowed", 405)

            return None  # Let handle_post handle it

        def handle_post(
            self, path: str, body: Dict[str, Any], handler: Any
        ) -> Optional[HandlerResult]:
            """Handle POST requests (webhooks)."""
            # Rate limit check
            if _chat_limiter is not None:
                client_ip = get_client_ip(handler)
                if not _chat_limiter.is_allowed(client_ip):
                    logger.warning(f"Rate limit exceeded for chat webhook: {client_ip}")
                    return error_response("Rate limit exceeded. Please try again later.", 429)

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
            elif "/telegram/" in path:
                platform = "telegram"
            elif "/whatsapp/" in path:
                platform = "whatsapp"
            else:
                # Try header-based detection first
                platform = self.router.detect_platform(headers)

                # Fall back to body-based detection if headers weren't definitive
                if not platform:
                    platform = self.router.detect_platform_from_body(headers, raw_body)
                    if platform:
                        logger.info(f"Detected platform '{platform}' from body structure")

            if not platform:
                logger.warning(
                    "Could not determine chat platform from headers or body. "
                    "Use platform-specific endpoint (e.g., /api/chat/telegram/webhook) "
                    "or configure webhook signatures."
                )
                return error_response(
                    "Could not determine platform. Use platform-specific endpoint.",
                    400,
                )

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


def _create_decision_router_debate_starter():
    """
    Create a debate starter that uses DecisionRouter for unified routing.

    This provides caching, deduplication, and metrics for all debates started
    from chat platforms.
    """

    async def debate_starter(
        topic: str,
        platform: str,
        channel: str,
        user: str,
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            from aragora.core import (
                DecisionRequest,
                DecisionType,
                InputSource,
                RequestContext,
                ResponseChannel,
                get_decision_router,
            )

            # Map platform to InputSource
            source_map = {
                "slack": InputSource.SLACK,
                "discord": InputSource.DISCORD,
                "teams": InputSource.TEAMS,
                "telegram": InputSource.TELEGRAM,
                "whatsapp": InputSource.WHATSAPP,
                "google_chat": InputSource.GOOGLE_CHAT,
            }

            # Create ResponseChannel with platform string
            response_channel = ResponseChannel(
                platform=platform,
                channel_id=channel,
                user_id=user,
            )

            # Create request context
            context = RequestContext(
                user_id=user,
                session_id=f"{platform}:{channel}",
            )

            request = DecisionRequest(
                content=topic,
                decision_type=DecisionType.DEBATE,
                source=source_map.get(platform, InputSource.HTTP_API),
                response_channels=[response_channel],
                context=context,
            )

            router = get_decision_router()
            result = await router.route(request)

            # Extract debate_id from debate_result if available
            debate_id = result.request_id
            if result.debate_result and hasattr(result.debate_result, "debate_id"):
                debate_id = result.debate_result.debate_id or debate_id

            return {
                "debate_id": debate_id or "pending",
                "status": "completed" if result.success else "failed",
                "topic": topic,
                "answer": result.answer,
                "confidence": result.confidence,
            }
        except ImportError:
            logger.debug("DecisionRouter not available, returning minimal response")
            return {"debate_id": "pending", "topic": topic}
        except Exception as e:
            logger.error(f"DecisionRouter debate start failed: {e}")
            raise

    return debate_starter


def get_webhook_router(
    event_handler: Optional[Callable[..., Any]] = None,
    debate_starter: Optional[Callable[..., Any]] = None,
    use_decision_router: bool = True,
) -> ChatWebhookRouter:
    """
    Get or create the webhook router singleton.

    Args:
        event_handler: Custom event handler callback
        debate_starter: Custom debate starter callback (overrides use_decision_router)
        use_decision_router: If True and no debate_starter provided, use DecisionRouter
            for unified routing with caching, deduplication, and metrics

    Returns:
        ChatWebhookRouter singleton instance
    """
    global _router
    if _router is None:
        # Use DecisionRouter-based debate starter by default if not provided
        if debate_starter is None and use_decision_router:
            debate_starter = _create_decision_router_debate_starter()

        _router = ChatWebhookRouter(event_handler, debate_starter)
    return _router
