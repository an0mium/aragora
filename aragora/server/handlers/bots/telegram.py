"""
Telegram Bot webhook handler.

Handles Telegram Bot API webhook updates for bidirectional chat.

Endpoints:
- POST /api/bots/telegram/webhook - Handle Telegram updates
- POST /api/bots/telegram/webhook/{token} - Token-verified webhook
- GET  /api/bots/telegram/status - Get bot status

Environment Variables:
- TELEGRAM_BOT_TOKEN - Bot API token from @BotFather
- TELEGRAM_WEBHOOK_SECRET - Optional secret for webhook URL verification
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any, Dict, Optional

from aragora.audit.unified import audit_data, audit_security
from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.secure import SecureHandler, ForbiddenError, UnauthorizedError
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# RBAC permission for bot configuration endpoints
BOTS_READ_PERMISSION = "bots:read"

# Environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")

# Compute expected webhook token from bot token (Telegram recommendation)
TELEGRAM_WEBHOOK_TOKEN = ""
if TELEGRAM_BOT_TOKEN:
    TELEGRAM_WEBHOOK_TOKEN = hashlib.sha256(TELEGRAM_BOT_TOKEN.encode()).hexdigest()[:32]


def _verify_telegram_secret(secret_token: str) -> bool:
    """Verify Telegram X-Telegram-Bot-Api-Secret-Token header.

    Telegram sends this header if you configured a secret_token when setting the webhook.
    See: https://core.telegram.org/bots/api#setwebhook
    """
    if not TELEGRAM_WEBHOOK_SECRET:
        # No secret configured, allow all
        return True

    return hmac.compare_digest(secret_token, TELEGRAM_WEBHOOK_SECRET)


def _verify_webhook_token(token: str) -> bool:
    """Verify token in webhook URL path.

    Alternative to secret header - embed token in URL.
    """
    if not TELEGRAM_WEBHOOK_TOKEN:
        return True

    return hmac.compare_digest(token, TELEGRAM_WEBHOOK_TOKEN)


class TelegramHandler(SecureHandler):
    """Handler for Telegram Bot API webhook endpoints.

    RBAC Protected:
    - bots:read - required for status endpoint

    Note: Webhook endpoints are authenticated via Telegram's secret token,
    not RBAC, since they are called by Telegram servers directly.
    """

    ROUTES = [
        "/api/v1/bots/telegram/webhook",
        "/api/v1/bots/telegram/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Also handle /api/bots/telegram/webhook/{token}
        if path.startswith("/api/v1/bots/telegram/webhook/"):
            return True
        return False

    @rate_limit(rpm=60)
    async def handle(  # type: ignore[override]
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Telegram GET requests with RBAC for status endpoint."""
        if path == "/api/v1/bots/telegram/status":
            # RBAC: Require authentication and bots:read permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, BOTS_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning(f"Telegram status access denied: {e}")
                return error_response(str(e), 403)
            return self._get_status()

        return None

    @rate_limit(rpm=120)
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests (webhook updates)."""
        if path == "/api/v1/bots/telegram/webhook":
            return self._handle_webhook(handler)

        # Handle /api/bots/telegram/webhook/{token}
        if path.startswith("/api/v1/bots/telegram/webhook/"):
            token = path.split("/")[-1]
            if not _verify_webhook_token(token):
                logger.warning("Telegram webhook token verification failed")
                audit_security(
                    event_type="telegram_webhook_auth_failed",
                    actor_id="unknown",
                    resource_type="telegram_webhook",
                    resource_id="token_path",
                )
                return error_response("Unauthorized", 401)
            return self._handle_webhook(handler, skip_secret_check=True)

        return None

    def _get_status(self) -> HandlerResult:
        """Get Telegram bot status."""
        return json_response(
            {
                "platform": "telegram",
                "enabled": bool(TELEGRAM_BOT_TOKEN),
                "token_configured": bool(TELEGRAM_BOT_TOKEN),
                "webhook_secret_configured": bool(TELEGRAM_WEBHOOK_SECRET),
                "webhook_token": (
                    TELEGRAM_WEBHOOK_TOKEN[:8] + "..." if TELEGRAM_WEBHOOK_TOKEN else None
                ),
            }
        )

    def _handle_webhook(self, handler: Any, skip_secret_check: bool = False) -> HandlerResult:
        """Handle Telegram webhook updates.

        Telegram sends updates as JSON to this endpoint when messages are received.
        See: https://core.telegram.org/bots/api#update
        """
        try:
            # Verify secret token header if configured
            if not skip_secret_check:
                secret_token = handler.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
                if not _verify_telegram_secret(secret_token):
                    logger.warning("Telegram secret token verification failed")
                    audit_security(
                        event_type="telegram_webhook_auth_failed",
                        actor_id="unknown",
                        resource_type="telegram_webhook",
                        resource_id="secret_header",
                    )
                    return error_response("Unauthorized", 401)

            # Read body
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            # Parse update
            try:
                update = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Telegram update: {e}")
                return error_response("Invalid JSON", 400)

            update_id = update.get("update_id")
            logger.debug(f"Telegram update received: {update_id}")

            # Route update types
            if "message" in update:
                return self._handle_message(update["message"])
            elif "callback_query" in update:
                return self._handle_callback_query(update["callback_query"])
            elif "inline_query" in update:
                return self._handle_inline_query(update["inline_query"])
            elif "edited_message" in update:
                return self._handle_message(update["edited_message"], edited=True)
            else:
                # Acknowledge unknown update types
                logger.debug(f"Unhandled Telegram update type: {list(update.keys())}")
                return json_response({"ok": True})

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Telegram update: {e}")
            return error_response("Invalid JSON payload", 400)
        except Exception as e:
            logger.exception(f"Unexpected Telegram webhook error: {e}")
            # Always return 200 to prevent Telegram from retrying
            return json_response({"ok": False, "error": str(e)[:100]})

    def _handle_message(self, message: Dict[str, Any], edited: bool = False) -> HandlerResult:
        """Handle incoming Telegram message."""
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        chat_type = chat.get("type", "private")

        from_user = message.get("from", {})
        user_id = from_user.get("id")
        username = from_user.get("username", from_user.get("first_name", "unknown"))

        text = message.get("text", "")

        logger.info(f"Telegram message from {username} in {chat_type}: {text[:50]}...")

        # Check for bot commands
        entities = message.get("entities", [])
        for entity in entities:
            if entity.get("type") == "bot_command" and entity.get("offset", 0) == 0:
                command = text[entity["offset"] : entity["offset"] + entity["length"]]
                args = text[entity["offset"] + entity["length"] :].strip()
                return self._handle_command(command, args, chat_id, user_id, message)

        # Handle regular message (could be a debate input)
        # For now, acknowledge receipt
        return json_response({"ok": True, "handled": "message"})

    def _handle_command(
        self,
        command: str,
        args: str,
        chat_id: int,
        user_id: int,
        message: Dict[str, Any],
    ) -> HandlerResult:
        """Handle Telegram bot command."""
        command = command.lower().lstrip("/")

        logger.info(f"Telegram command: /{command} {args[:50]}...")

        # Route commands
        if command == "start":
            return self._cmd_start(chat_id, user_id)
        elif command == "help":
            return self._cmd_help(chat_id)
        elif command == "debate":
            return self._cmd_debate(chat_id, user_id, args)
        elif command == "status":
            return self._cmd_status(chat_id)
        elif command in ("aragora", "ask"):
            return self._cmd_debate(chat_id, user_id, args)
        else:
            return self._cmd_unknown(chat_id, command)

    def _handle_callback_query(self, callback: Dict[str, Any]) -> HandlerResult:
        """Handle callback query (inline button press)."""
        callback_id = callback.get("id")
        data = callback.get("data", "")

        from_user = callback.get("from", {})
        user_id = from_user.get("id")

        logger.info(f"Telegram callback from {user_id}: {data}")

        # Parse callback data (format: action:param1:param2)
        parts = data.split(":")
        action = parts[0] if parts else ""

        if action == "vote":
            # Handle vote callback
            debate_id = parts[1] if len(parts) > 1 else ""
            vote_option = parts[2] if len(parts) > 2 else ""
            return self._handle_vote(callback_id, user_id, debate_id, vote_option)

        # Default: acknowledge callback
        return json_response({"ok": True, "callback_handled": True})

    def _handle_inline_query(self, query: Dict[str, Any]) -> HandlerResult:
        """Handle inline query (@bot query)."""
        query.get("id")
        query_text = query.get("query", "")

        logger.debug(f"Telegram inline query: {query_text[:50]}...")

        # For now, return empty results
        return json_response({"ok": True})

    def _handle_vote(
        self, callback_id: str, user_id: int, debate_id: str, vote_option: str
    ) -> HandlerResult:
        """Handle vote on debate outcome."""
        logger.info(f"Vote from {user_id} on {debate_id}: {vote_option}")

        try:
            from aragora.memory.consensus import ConsensusStore  # type: ignore[attr-defined]

            # Record the vote
            store = ConsensusStore()
            store.record_vote(
                debate_id=debate_id,
                user_id=f"telegram:{user_id}",
                vote=vote_option,
                source="telegram",
            )

            # Acknowledge the callback
            self._answer_callback_query(callback_id, f"Vote recorded: {vote_option}")

            logger.info(f"Vote recorded from Telegram user {user_id} on {debate_id}: {vote_option}")
            audit_data(
                user_id=f"telegram:{user_id}",
                resource_type="debate_vote",
                resource_id=debate_id,
                action="create",
                vote_option=vote_option,
                platform="telegram",
            )
            return json_response({"ok": True, "vote_recorded": True})

        except ImportError:
            logger.warning("ConsensusStore not available for vote recording")
            self._answer_callback_query(callback_id, "Vote acknowledged")
            return json_response({"ok": True, "vote_recorded": False})
        except Exception as e:
            logger.error(f"Failed to record vote: {e}")
            self._answer_callback_query(callback_id, "Error recording vote")
            return json_response({"ok": False, "error": str(e)[:100]})

    def _answer_callback_query(self, callback_id: str, text: str) -> None:
        """Answer a callback query (acknowledge button press)."""
        if not TELEGRAM_BOT_TOKEN:
            return

        try:
            import httpx

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
            data = {
                "callback_query_id": callback_id,
                "text": text,
            }

            with httpx.Client(timeout=10.0) as client:
                client.post(url, json=data)

        except Exception as e:
            logger.error(f"Failed to answer callback query: {e}")

    # Command implementations

    def _cmd_start(self, chat_id: int, user_id: int) -> HandlerResult:
        """Handle /start command."""
        self._send_message(
            chat_id,
            "Welcome to Aragora - Control plane for multi-agent vetted decisionmaking!\n\n"
            "I orchestrate 15+ AI models (Claude, GPT, Gemini, Grok and more) "
            "to debate and deliver defensible decisions.\n\n"
            "Commands:\n"
            "/debate <question> - Start a multi-agent vetted decisionmaking\n"
            "/status - Check system status\n"
            "/help - Show this help\n\n"
            "Or just send me a question and I'll deliberate!",
        )
        return json_response({"ok": True})

    def _cmd_help(self, chat_id: int) -> HandlerResult:
        """Handle /help command."""
        self._send_message(
            chat_id,
            "Aragora Commands:\n\n"
            "/debate <question> - Start a multi-agent debate\n"
            "/ask <question> - Alias for /debate\n"
            "/status - Check Aragora system status\n"
            "/help - Show this message\n\n"
            "Example:\n"
            "/debate Should we use microservices or a monolith?",
        )
        return json_response({"ok": True})

    def _cmd_debate(self, chat_id: int, user_id: int, topic: str) -> HandlerResult:
        """Handle /debate command."""
        if not topic.strip():
            self._send_message(
                chat_id,
                "Please provide a topic. Example:\n/debate Is Python better than JavaScript?",
            )
            return json_response({"ok": True})

        # Start debate via queue system
        debate_id = self._start_debate_async(chat_id, user_id, topic)

        self._send_message(
            chat_id,
            f"Starting debate on:\n\n{topic[:200]}\n\n"
            "I'll notify you when the AI agents reach consensus. "
            f"Debate ID: {debate_id[:8]}...",
        )

        logger.info(f"Debate requested from Telegram user {user_id}: {topic[:100]}")
        return json_response({"ok": True, "debate_started": True, "debate_id": debate_id})

    def _start_debate_async(self, chat_id: int, user_id: int, topic: str) -> str:
        """Start a debate asynchronously via the DecisionRouter.

        Uses the unified DecisionRouter for:
        - Deduplication (prevents duplicate debates for same topic/user)
        - Caching (returns cached results if available)
        - Metrics and logging
        - Origin registration for result routing
        """
        import asyncio
        import uuid

        debate_id = str(uuid.uuid4())

        async def route_debate():
            try:
                from aragora.core import (
                    DecisionRequest,
                    DecisionType,
                    InputSource,
                    RequestContext,
                    ResponseChannel,
                    get_decision_router,
                )

                # Create response channel for result routing
                response_channel = ResponseChannel(
                    platform="telegram",
                    channel_id=str(chat_id),
                    user_id=str(user_id),
                )

                # Create request context
                context = RequestContext(
                    user_id=str(user_id),
                    session_id=f"telegram:{chat_id}",
                )

                # Create decision request
                request = DecisionRequest(
                    content=topic,
                    decision_type=DecisionType.DEBATE,
                    source=InputSource.TELEGRAM,
                    response_channels=[response_channel],
                    context=context,
                )

                # Route through DecisionRouter (handles origin registration, deduplication, caching)
                router = get_decision_router()
                result = await router.route(request)

                if result.debate_id:
                    logger.info(f"DecisionRouter started debate {result.debate_id} from Telegram")
                    return result.debate_id
                return debate_id

            except ImportError:
                logger.debug("DecisionRouter not available, falling back to queue system")
                return await self._fallback_queue_debate(chat_id, user_id, topic, debate_id)
            except Exception as e:
                logger.error(f"DecisionRouter failed: {e}, falling back to queue system")
                return await self._fallback_queue_debate(chat_id, user_id, topic, debate_id)

        # Run async routing
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as task and return preliminary ID
                asyncio.create_task(route_debate())
                return debate_id
            else:
                return asyncio.run(route_debate())
        except RuntimeError:
            return asyncio.run(route_debate())

    async def _fallback_queue_debate(
        self, chat_id: int, user_id: int, topic: str, debate_id: str
    ) -> str:
        """Fallback to direct queue enqueue if DecisionRouter unavailable."""
        # Register origin for result routing
        try:
            from aragora.server.debate_origin import register_debate_origin

            register_debate_origin(
                debate_id=debate_id,
                platform="telegram",
                channel_id=str(chat_id),
                user_id=str(user_id),
                metadata={"topic": topic},
            )
        except Exception as e:
            logger.warning(f"Failed to register debate origin: {e}")

        try:
            from aragora.queue import create_debate_job, create_redis_queue

            job = create_debate_job(
                question=topic,
                agents=None,
                rounds=3,
                consensus="majority",
                protocol="standard",
                user_id=f"telegram:{user_id}",
                webhook_url=None,
            )

            queue = await create_redis_queue()
            await queue.enqueue(job)
            logger.info(f"Debate job enqueued via fallback: {job.job_id}")  # type: ignore[attr-defined]
            return job.job_id  # type: ignore[attr-defined]

        except ImportError:
            logger.warning("Queue system not available, using direct execution")
            return self._run_debate_direct(chat_id, user_id, topic, debate_id)
        except Exception as e:
            logger.error(f"Failed to enqueue debate: {e}")
            return debate_id

    def _run_debate_direct(self, chat_id: int, user_id: int, topic: str, debate_id: str) -> str:
        """Run debate directly without queue (fallback)."""
        import asyncio
        import threading

        def run_in_thread():
            try:
                from aragora.debate.orchestrator import Arena
                from aragora import Environment, DebateProtocol
                from aragora.agents.cli_agents import get_default_agents

                async def execute():
                    env = Environment(task=topic)
                    protocol = DebateProtocol(rounds=3, consensus="majority")
                    agents = get_default_agents()[:3]  # Use first 3 agents
                    arena = Arena(env, agents, protocol)
                    result = await arena.run()

                    # Send result back to user
                    if result and result.consensus_reached:
                        self._send_message(
                            chat_id,
                            f"Debate Complete!\n\n"
                            f"Topic: {topic[:100]}\n\n"
                            f"Consensus: {result.final_answer[:500]}\n\n"
                            f"Confidence: {result.confidence:.0%}",
                        )
                    else:
                        self._send_message(
                            chat_id,
                            f"Debate Complete!\n\n"
                            f"Topic: {topic[:100]}\n\n"
                            "No consensus was reached. The agents had differing views.",
                        )

                asyncio.run(execute())

            except Exception as e:
                logger.error(f"Direct debate execution failed: {e}")
                self._send_message(chat_id, f"Debate failed: {str(e)[:100]}")

        # Run in background thread to not block webhook response
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        return debate_id

    def _cmd_status(self, chat_id: int) -> HandlerResult:
        """Handle /status command."""
        self._send_message(
            chat_id,
            "Aragora Status: Online\n\n"
            "Available AI models:\n"
            "- Claude (Anthropic)\n"
            "- GPT-4 (OpenAI)\n"
            "- Gemini (Google)\n"
            "- Grok (xAI)\n"
            "- Mistral\n"
            "- DeepSeek\n"
            "- Qwen\n\n"
            "Ready for debates!",
        )
        return json_response({"ok": True})

    def _cmd_unknown(self, chat_id: int, command: str) -> HandlerResult:
        """Handle unknown command."""
        self._send_message(
            chat_id, f"Unknown command: /{command}\n\nUse /help to see available commands."
        )
        return json_response({"ok": True})

    def _send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown") -> None:
        """Send a message via Telegram Bot API.

        This is a fire-and-forget operation for webhook responses.
        For reliable sending, use the TelegramConnector.
        """
        if not TELEGRAM_BOT_TOKEN:
            logger.warning("Cannot send Telegram message: TELEGRAM_BOT_TOKEN not configured")
            return

        try:
            import httpx

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }

            # Fire and forget - use async in production
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=data)
                if not response.is_success:
                    logger.warning(f"Telegram send failed: {response.status_code}")

        except ImportError:
            logger.warning("httpx not available for Telegram messaging")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
