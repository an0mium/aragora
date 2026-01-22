"""
Telegram Bot integration endpoint handlers.

Endpoints:
- POST /api/integrations/telegram/webhook - Handle Telegram webhook updates
- GET  /api/integrations/telegram/status  - Get integration status
- POST /api/integrations/telegram/set-webhook - Configure webhook URL

Environment Variables:
- TELEGRAM_BOT_TOKEN - Required for all API calls
- TELEGRAM_WEBHOOK_SECRET - Optional secret for webhook verification

Telegram Bot Commands:
- /start - Welcome message
- /help - Show available commands
- /debate <topic> - Start a multi-agent debate
- /gauntlet <statement> - Run adversarial validation
- /status - Get system status
- /agents - List available agents
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from typing import Any, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug(f"Task {task_name} was cancelled")
    elif task.exception():
        exc = task.exception()
        logger.error(f"Task {task_name} failed with exception: {exc}", exc_info=exc)


def create_tracked_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(lambda t: _handle_task_exception(t, name))
    return task


from ..base import (
    BaseHandler,
    HandlerResult,
    auto_error_response,
    error_response,
    json_response,
)
from ..utils.rate_limit import rate_limit
from .telemetry import (
    record_api_call,
    record_api_latency,
    record_command,
    record_debate_completed,
    record_debate_failed,
    record_debate_started,
    record_error,
    record_gauntlet_completed,
    record_gauntlet_failed,
    record_gauntlet_started,
    record_message,
    record_vote,
    record_webhook_latency,
    record_webhook_request,
)
from .chat_events import (
    emit_command_received,
    emit_debate_completed,
    emit_debate_started,
    emit_gauntlet_completed,
    emit_gauntlet_started,
    emit_message_received,
    emit_vote_received,
)

# TTS support
TTS_VOICE_ENABLED = os.environ.get("TELEGRAM_TTS_ENABLED", "false").lower() == "true"

# Environment variables for Telegram integration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
TELEGRAM_API_BASE = "https://api.telegram.org/bot"


class TelegramHandler(BaseHandler):
    """Handler for Telegram Bot integration endpoints."""

    ROUTES = [
        "/api/v1/integrations/telegram/webhook",
        "/api/v1/integrations/telegram/status",
        "/api/v1/integrations/telegram/set-webhook",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route Telegram requests to appropriate methods."""
        logger.debug(f"Telegram request: {path}")

        if path == "/api/v1/integrations/telegram/status":
            return self._get_status()

        if path == "/api/v1/integrations/telegram/set-webhook":
            if handler.command != "POST":
                return error_response("Method not allowed", 405)
            return self._set_webhook(handler)

        if path == "/api/v1/integrations/telegram/webhook":
            if handler.command != "POST":
                return error_response("Method not allowed", 405)

            # Verify webhook secret if configured
            if TELEGRAM_WEBHOOK_SECRET and not self._verify_secret(handler):
                logger.warning("Telegram webhook secret verification failed")
                return error_response("Unauthorized", 401)

            return self._handle_webhook(handler)

        return error_response("Not found", 404)

    def handle_post(self, path: str, body: Dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """Handle POST requests."""
        return self.handle(path, {}, handler)

    def _verify_secret(self, handler: Any) -> bool:
        """Verify Telegram webhook secret token.

        Telegram supports a secret_token parameter in setWebhook that is sent
        in the X-Telegram-Bot-Api-Secret-Token header.
        """
        if not TELEGRAM_WEBHOOK_SECRET:
            return True

        try:
            secret_header = handler.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            return hmac.compare_digest(secret_header, TELEGRAM_WEBHOOK_SECRET)
        except Exception as e:
            logger.warning(f"Error verifying Telegram secret: {e}")
            return False

    def _get_status(self) -> HandlerResult:
        """Get Telegram integration status."""
        return json_response(
            {
                "enabled": bool(TELEGRAM_BOT_TOKEN),
                "bot_token_configured": bool(TELEGRAM_BOT_TOKEN),
                "webhook_secret_configured": bool(TELEGRAM_WEBHOOK_SECRET),
            }
        )

    @auto_error_response("set telegram webhook")
    def _set_webhook(self, handler: Any) -> HandlerResult:
        """Configure Telegram webhook URL."""
        if not TELEGRAM_BOT_TOKEN:
            return error_response("Telegram bot token not configured", 500)

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            webhook_url = data.get("url")
            if not webhook_url:
                return error_response("webhook url is required", 400)

            # Set webhook via Telegram API
            create_tracked_task(
                self._set_webhook_async(webhook_url),
                name="telegram-set-webhook",
            )

            return json_response({"status": "webhook configuration queued"})

        except json.JSONDecodeError:
            return error_response("Invalid JSON body", 400)
        except Exception as e:
            logger.error(f"Failed to set webhook: {e}")
            return error_response(f"Failed to set webhook: {str(e)[:100]}", 500)

    async def _set_webhook_async(self, webhook_url: str) -> None:
        """Set Telegram webhook via API."""
        import aiohttp

        try:
            url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/setWebhook"
            payload = {"url": webhook_url}

            if TELEGRAM_WEBHOOK_SECRET:
                payload["secret_token"] = TELEGRAM_WEBHOOK_SECRET

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    result = await response.json()
                    if result.get("ok"):
                        logger.info(f"Telegram webhook set to: {webhook_url}")
                    else:
                        logger.error(f"Failed to set webhook: {result}")
        except Exception as e:
            logger.error(f"Error setting Telegram webhook: {e}")

    @auto_error_response("handle telegram webhook")
    @rate_limit(rpm=100, limiter_name="telegram_webhook")
    def _handle_webhook(self, handler: Any) -> HandlerResult:
        """Handle incoming Telegram webhook updates.

        Update types:
        - message: Text message from user
        - callback_query: Button click callback
        - edited_message: Edited message
        - inline_query: Inline bot query
        """
        import time

        start_time = time.time()
        status = "success"
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            update = json.loads(body)

            logger.debug(f"Telegram update received: {update.get('update_id')}")

            # Handle different update types
            if "message" in update:
                return self._handle_message(update["message"])
            elif "callback_query" in update:
                record_message("telegram", "callback")
                return self._handle_callback_query(update["callback_query"])
            elif "edited_message" in update:
                # Ignore edited messages for now
                record_message("telegram", "edited")
                return json_response({"ok": True})
            elif "inline_query" in update:
                record_message("telegram", "inline")
                return self._handle_inline_query(update["inline_query"])

            # Acknowledge unknown updates
            return json_response({"ok": True})

        except json.JSONDecodeError:
            logger.warning("Invalid JSON in Telegram webhook")
            status = "error"
            record_error("telegram", "json_parse")
            return json_response({"ok": True})
        except Exception as e:
            logger.error(f"Error handling Telegram webhook: {e}", exc_info=True)
            status = "error"
            record_error("telegram", "unknown")
            return json_response({"ok": True})
        finally:
            latency = time.time() - start_time
            record_webhook_request("telegram", status)
            record_webhook_latency("telegram", latency)

    def _handle_message(self, message: Dict[str, Any]) -> HandlerResult:
        """Handle incoming text message."""
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "").strip()
        user = message.get("from", {})
        user_id = user.get("id")
        username = user.get("username", "unknown")

        if not chat_id or not text:
            return json_response({"ok": True})

        logger.info(f"Telegram message from {username} ({user_id}): {text[:50]}...")

        # Parse bot commands
        if text.startswith("/"):
            record_message("telegram", "command")
            return self._handle_command(chat_id, user_id, username, text)

        record_message("telegram", "text")

        # Emit webhook event for message received
        emit_message_received(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            message_text=text,
            message_type="text",
        )

        # Handle regular messages as questions/topics
        if len(text) > 10:
            response = (
                f'I received: "{text[:50]}..."\n\n'
                "To start a debate on this topic, use:\n"
                f"/debate {text[:100]}"
            )
        else:
            response = "Send /help to see available commands."

        create_tracked_task(
            self._send_message_async(chat_id, response),
            name=f"telegram-reply-{chat_id}",
        )

        return json_response({"ok": True})

    def _handle_command(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        text: str,
    ) -> HandlerResult:
        """Handle bot commands."""
        # Parse command and arguments
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        # Remove @botname suffix if present
        if "@" in command:
            command = command.split("@")[0]
        args = parts[1] if len(parts) > 1 else ""

        # Record command metric (strip leading /)
        cmd_name = command[1:] if command.startswith("/") else command

        # Emit webhook event for command received
        emit_command_received(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            command=cmd_name,
            args=args,
        )

        if command == "/start":
            record_command("telegram", "start")
            response = self._command_start(username)
        elif command == "/help":
            record_command("telegram", "help")
            response = self._command_help()
        elif command == "/status":
            record_command("telegram", "status")
            response = self._command_status()
        elif command == "/agents":
            record_command("telegram", "agents")
            response = self._command_agents()
        elif command == "/debate":
            record_command("telegram", "debate")
            return self._command_debate(chat_id, user_id, username, args)
        elif command == "/gauntlet":
            record_command("telegram", "gauntlet")
            return self._command_gauntlet(chat_id, user_id, username, args)
        else:
            record_command("telegram", "unknown")
            response = f"Unknown command: {command}\nSend /help for available commands."

        create_tracked_task(
            self._send_message_async(chat_id, response),
            name=f"telegram-cmd-{command}-{chat_id}",
        )

        return json_response({"ok": True})

    def _command_start(self, username: str) -> str:
        """Handle /start command."""
        return (
            f"Welcome to Aragora, {username}!\n\n"
            "I can run multi-agent debates and adversarial validations.\n\n"
            "Commands:\n"
            "/debate <topic> - Start a debate\n"
            "/gauntlet <statement> - Stress-test a statement\n"
            "/status - System status\n"
            "/agents - List agents\n"
            "/help - Show this help"
        )

    def _command_help(self) -> str:
        """Handle /help command."""
        return (
            "*Aragora Bot Commands*\n\n"
            "/start - Welcome message\n"
            "/debate <topic> - Start a multi-agent debate on a topic\n"
            "/gauntlet <statement> - Run adversarial stress-test\n"
            "/status - Get system status\n"
            "/agents - List available agents\n"
            "/help - Show this help\n\n"
            "*Examples:*\n"
            "/debate Should AI be regulated?\n"
            "/gauntlet We should migrate to microservices"
        )

    def _command_status(self) -> str:
        """Handle /status command."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()
            return f"*Aragora Status*\n\nStatus: Online\nAgents: {len(agents)} registered"
        except Exception as e:
            logger.warning(f"Failed to get status: {e}")
            return "*Aragora Status*\n\nStatus: Online"

    def _command_agents(self) -> str:
        """Handle /agents command."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()

            if not agents:
                return "No agents registered yet."

            agents = sorted(agents, key=lambda a: getattr(a, "elo", 1500), reverse=True)

            lines = ["*Top Agents by ELO:*\n"]
            for i, agent in enumerate(agents[:10]):
                name = getattr(agent, "name", "Unknown")
                elo = getattr(agent, "elo", 1500)
                wins = getattr(agent, "wins", 0)
                medal = ["1.", "2.", "3."][i] if i < 3 else f"{i + 1}."
                lines.append(f"{medal} *{name}* - ELO: {elo:.0f} | Wins: {wins}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to list agents: {e}")
            return "Could not fetch agent list."

    def _command_debate(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        args: str,
    ) -> HandlerResult:
        """Handle /debate command."""
        if not args:
            response = "Please provide a topic.\n\nExample: /debate Should AI be regulated?"
            create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-debate-help-{chat_id}",
            )
            return json_response({"ok": True})

        topic = args.strip().strip("\"'")

        if len(topic) < 10:
            response = "Topic is too short. Please provide more detail."
            create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-debate-short-{chat_id}",
            )
            return json_response({"ok": True})

        if len(topic) > 500:
            response = "Topic is too long. Please limit to 500 characters."
            create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-debate-long-{chat_id}",
            )
            return json_response({"ok": True})

        # Send initial acknowledgment
        create_tracked_task(
            self._send_message_async(
                chat_id,
                f"*Starting debate on:*\n_{topic}_\n\nRequested by @{username}\nProcessing... (this may take a few minutes)",
                parse_mode="Markdown",
            ),
            name=f"telegram-debate-ack-{chat_id}",
        )

        # Queue the debate asynchronously
        create_tracked_task(
            self._run_debate_async(chat_id, user_id, username, topic),
            name=f"telegram-debate-{topic[:30]}",
        )

        return json_response({"ok": True})

    async def _run_debate_async(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        topic: str,
        message_id: Optional[int] = None,
    ) -> None:
        """Run debate asynchronously and send result to chat."""
        record_debate_started("telegram")
        try:
            from aragora import Arena, DebateProtocol, Environment
            from aragora.agents import get_agents_by_names  # type: ignore[attr-defined]

            # Register debate origin for tracking and potential async routing
            try:
                from aragora.server.debate_origin import register_debate_origin
                import uuid

                debate_id = f"tg-{chat_id}-{uuid.uuid4().hex[:8]}"
                register_debate_origin(
                    debate_id=debate_id,
                    platform="telegram",
                    channel_id=str(chat_id),
                    user_id=str(user_id),
                    message_id=str(message_id) if message_id else None,
                    metadata={"username": username, "topic": topic},
                )
                logger.debug(f"Registered debate origin: {debate_id}")
            except ImportError:
                debate_id = None
                logger.debug("Debate origin tracking not available")

            # Emit webhook event for debate started
            emit_debate_started(
                platform="telegram",
                chat_id=str(chat_id),
                user_id=str(user_id),
                username=username,
                topic=topic,
                debate_id=debate_id,
            )

            env = Environment(task=f"Debate: {topic}")
            agents = get_agents_by_names(["anthropic-api", "openai-api"])
            protocol = DebateProtocol(
                rounds=3,
                consensus="majority",
                convergence_detection=False,
                early_stopping=False,
            )

            if not agents:
                await self._send_message_async(
                    chat_id,
                    "Failed to start debate: No agents available",
                )
                record_debate_failed("telegram")
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            consensus_emoji = "+" if result.consensus_reached else "-"
            confidence_pct = f"{result.confidence:.1%}"

            response = (
                f"*Debate Complete!* {consensus_emoji}\n\n"
                f"*Topic:* {topic[:100]}{'...' if len(topic) > 100 else ''}\n\n"
                f"*Consensus:* {'Yes' if result.consensus_reached else 'No'}\n"
                f"*Confidence:* {confidence_pct}\n"
                f"*Rounds:* {result.rounds_used}\n"
                f"*Agents:* {len(agents)}\n\n"
                f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion reached'}{'...' if result.final_answer and len(result.final_answer) > 500 else ''}\n\n"
                f"_Debate ID: {result.id}_\n"
                f"_Requested by @{username}_"
            )

            # Send with inline keyboard for voting
            keyboard = {
                "inline_keyboard": [
                    [
                        {"text": "Agree", "callback_data": f"vote:{result.id}:agree"},
                        {"text": "Disagree", "callback_data": f"vote:{result.id}:disagree"},
                    ],
                    [
                        {"text": "View Details", "callback_data": f"details:{result.id}"},
                    ],
                ]
            }

            await self._send_message_async(
                chat_id,
                response,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )

            # Mark debate result as sent for origin tracking
            if debate_id:
                try:
                    from aragora.server.debate_origin import mark_result_sent

                    mark_result_sent(debate_id)
                except ImportError:
                    pass

            # Send voice summary if TTS is enabled
            if TTS_VOICE_ENABLED:
                await self._send_voice_summary(
                    chat_id,
                    topic,
                    result.final_answer,
                    result.consensus_reached,
                    result.confidence,
                    result.rounds_used,
                )

            # Emit webhook event for debate completed
            emit_debate_completed(
                platform="telegram",
                chat_id=str(chat_id),
                debate_id=result.id,
                topic=topic,
                consensus_reached=result.consensus_reached,
                confidence=result.confidence,
                rounds_used=result.rounds_used,
                final_answer=result.final_answer,
            )

            # Record successful debate completion
            record_debate_completed("telegram", result.consensus_reached)

        except Exception as e:
            logger.error(f"Telegram debate failed: {e}", exc_info=True)
            record_debate_failed("telegram")
            await self._send_message_async(
                chat_id,
                f"Debate failed: {str(e)[:100]}",
            )

    def _command_gauntlet(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        args: str,
    ) -> HandlerResult:
        """Handle /gauntlet command."""
        if not args:
            response = "Please provide a statement to stress-test.\n\nExample: /gauntlet We should migrate to microservices"
            create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-gauntlet-help-{chat_id}",
            )
            return json_response({"ok": True})

        statement = args.strip().strip("\"'")

        if len(statement) < 10:
            response = "Statement is too short. Please provide more detail."
            create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-gauntlet-short-{chat_id}",
            )
            return json_response({"ok": True})

        if len(statement) > 1000:
            response = "Statement is too long. Please limit to 1000 characters."
            create_tracked_task(
                self._send_message_async(chat_id, response),
                name=f"telegram-gauntlet-long-{chat_id}",
            )
            return json_response({"ok": True})

        # Send initial acknowledgment
        create_tracked_task(
            self._send_message_async(
                chat_id,
                f"*Running Gauntlet stress-test on:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\nRequested by @{username}\nRunning adversarial validation...",
                parse_mode="Markdown",
            ),
            name=f"telegram-gauntlet-ack-{chat_id}",
        )

        # Queue the gauntlet asynchronously
        create_tracked_task(
            self._run_gauntlet_async(chat_id, user_id, username, statement),
            name=f"telegram-gauntlet-{statement[:30]}",
        )

        return json_response({"ok": True})

    async def _run_gauntlet_async(
        self,
        chat_id: int,
        user_id: int,
        username: str,
        statement: str,
    ) -> None:
        """Run gauntlet asynchronously and send result to chat."""
        import aiohttp

        record_gauntlet_started("telegram")

        # Emit webhook event for gauntlet started
        emit_gauntlet_started(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            statement=statement,
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8080/api/gauntlet/run",
                    json={
                        "statement": statement,
                        "intensity": "medium",
                        "metadata": {
                            "source": "telegram",
                            "chat_id": chat_id,
                            "user_id": user_id,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    data = await resp.json()

                    if resp.status != 200:
                        await self._send_message_async(
                            chat_id,
                            f"Gauntlet failed: {data.get('error', 'Unknown error')}",
                        )
                        record_gauntlet_failed("telegram")
                        return

                    run_id = data.get("run_id", "unknown")
                    score = data.get("score", 0)
                    passed = data.get("passed", False)
                    vulnerabilities = data.get("vulnerabilities", [])

                    status_emoji = "PASSED" if passed else "FAILED"
                    score_pct = f"{score:.1%}"

                    response = (
                        f"*Gauntlet Results* {status_emoji}\n\n"
                        f"*Statement:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\n"
                        f"*Score:* {score_pct}\n"
                        f"*Status:* {'Passed' if passed else 'Failed'}\n"
                        f"*Vulnerabilities:* {len(vulnerabilities)}\n"
                    )

                    if vulnerabilities:
                        response += "\n*Issues Found:*\n"
                        for v in vulnerabilities[:5]:
                            desc = v.get("description", "Unknown issue")[:100]
                            response += f"- {desc}\n"
                        if len(vulnerabilities) > 5:
                            response += f"_...and {len(vulnerabilities) - 5} more_\n"

                    response += f"\n_Run ID: {run_id}_\n_Requested by @{username}_"

                    await self._send_message_async(
                        chat_id,
                        response,
                        parse_mode="Markdown",
                    )

                    # Emit webhook event for gauntlet completed
                    emit_gauntlet_completed(
                        platform="telegram",
                        chat_id=str(chat_id),
                        gauntlet_id=run_id,
                        statement=statement,
                        verdict="passed" if passed else "failed",
                        confidence=score,
                        challenges_passed=len(
                            [v for v in vulnerabilities if not v.get("critical", False)]
                        ),
                        challenges_total=len(vulnerabilities) + 1,
                    )

                    # Record successful gauntlet completion
                    record_gauntlet_completed("telegram", passed)

        except Exception as e:
            logger.error(f"Telegram gauntlet failed: {e}", exc_info=True)
            record_gauntlet_failed("telegram")
            await self._send_message_async(
                chat_id,
                f"Gauntlet failed: {str(e)[:100]}",
            )

    def _handle_callback_query(self, callback: Dict[str, Any]) -> HandlerResult:
        """Handle inline keyboard button clicks."""
        callback_id = callback.get("id")
        data = callback.get("data", "")
        user = callback.get("from", {})
        user_id = user.get("id")
        username = user.get("username", "unknown")
        message = callback.get("message", {})
        chat_id = message.get("chat", {}).get("id")

        logger.info(f"Telegram callback from {username}: {data}")

        # Parse callback data
        parts = data.split(":")
        action = parts[0] if parts else ""

        if action == "vote" and len(parts) >= 3:
            debate_id = parts[1]
            vote_option = parts[2]
            return self._handle_vote(
                callback_id, chat_id, user_id, username, debate_id, vote_option
            )
        elif action == "details" and len(parts) >= 2:
            debate_id = parts[1]
            return self._handle_view_details(callback_id, chat_id, debate_id)

        # Answer callback to remove loading state
        create_tracked_task(
            self._answer_callback_async(callback_id, "Action received"),
            name=f"telegram-callback-ack-{callback_id}",
        )

        return json_response({"ok": True})

    def _handle_vote(
        self,
        callback_id: str,
        chat_id: int,
        user_id: int,
        username: str,
        debate_id: str,
        vote_option: str,
    ) -> HandlerResult:
        """Handle vote callback."""
        logger.info(f"Vote received: {debate_id} -> {vote_option} from {username}")

        # Emit webhook event for vote received
        emit_vote_received(
            platform="telegram",
            chat_id=str(chat_id),
            user_id=str(user_id),
            username=username,
            debate_id=debate_id,
            vote=vote_option,
        )

        # Record vote metrics
        record_vote("telegram", vote_option)

        # Record vote in storage
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db and hasattr(db, "record_vote"):
                db.record_vote(
                    debate_id=debate_id,
                    voter_id=f"telegram:{user_id}",
                    vote=vote_option,
                    source="telegram",
                )
        except Exception as e:
            logger.warning(f"Failed to record vote: {e}")

        emoji = "+" if vote_option == "agree" else "-"
        create_tracked_task(
            self._answer_callback_async(
                callback_id,
                f"{emoji} Your vote for '{vote_option}' has been recorded!",
                show_alert=True,
            ),
            name=f"telegram-vote-ack-{callback_id}",
        )

        return json_response({"ok": True})

    def _handle_view_details(
        self,
        callback_id: str,
        chat_id: int,
        debate_id: str,
    ) -> HandlerResult:
        """Handle view details callback."""
        debate_data = None
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db:
                debate_data = db.get(debate_id)
        except Exception as e:
            logger.warning(f"Failed to fetch debate: {e}")

        if not debate_data:
            create_tracked_task(
                self._answer_callback_async(
                    callback_id,
                    f"Debate {debate_id} not found",
                    show_alert=True,
                ),
                name=f"telegram-details-notfound-{callback_id}",
            )
            return json_response({"ok": True})

        task = debate_data.get("task", "Unknown")
        final_answer = debate_data.get("final_answer", "No conclusion")
        consensus = debate_data.get("consensus_reached", False)
        confidence = debate_data.get("confidence", 0)
        rounds_used = debate_data.get("rounds_used", 0)
        agents = debate_data.get("agents", [])

        agent_list = ", ".join(agents[:5]) if agents else "Unknown"
        if len(agents) > 5:
            agent_list += f" (+{len(agents) - 5} more)"

        response = (
            f"*Debate Details*\n\n"
            f"*Topic:*\n{task[:200]}{'...' if len(task) > 200 else ''}\n\n"
            f"*ID:* `{debate_id}`\n"
            f"*Consensus:* {'Yes' if consensus else 'No'}\n"
            f"*Confidence:* {confidence:.1%}\n"
            f"*Rounds:* {rounds_used}\n"
            f"*Agents:* {agent_list}\n\n"
            f"*Conclusion:*\n{final_answer[:500] if final_answer else 'No conclusion'}{'...' if final_answer and len(final_answer) > 500 else ''}"
        )

        # Answer callback and send details as new message
        create_tracked_task(
            self._answer_callback_async(callback_id, "Loading details..."),
            name=f"telegram-details-ack-{callback_id}",
        )

        create_tracked_task(
            self._send_message_async(chat_id, response, parse_mode="Markdown"),
            name=f"telegram-details-{chat_id}",
        )

        return json_response({"ok": True})

    def _handle_inline_query(self, query: Dict[str, Any]) -> HandlerResult:
        """Handle inline queries (@bot query)."""
        query_id = query.get("id")
        query_text = query.get("query", "").strip()

        if not query_text or len(query_text) < 5:
            results: List[Dict[str, Any]] = []
        else:
            # Provide inline results for debate topics
            results = [
                {
                    "type": "article",
                    "id": f"debate_{hash(query_text) % 10000}",
                    "title": f"Start debate: {query_text[:50]}...",
                    "description": "Click to start a multi-agent debate on this topic",
                    "input_message_content": {
                        "message_text": f"/debate {query_text}",
                    },
                },
                {
                    "type": "article",
                    "id": f"gauntlet_{hash(query_text) % 10000}",
                    "title": f"Stress-test: {query_text[:50]}...",
                    "description": "Click to run adversarial validation on this statement",
                    "input_message_content": {
                        "message_text": f"/gauntlet {query_text}",
                    },
                },
            ]

        create_tracked_task(
            self._answer_inline_query_async(query_id, results),
            name=f"telegram-inline-{query_id}",
        )

        return json_response({"ok": True})

    async def _send_message_async(
        self,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a message to Telegram chat."""
        import aiohttp
        import time

        if not TELEGRAM_BOT_TOKEN:
            logger.warning("Cannot send message: TELEGRAM_BOT_TOKEN not configured")
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload: Dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if reply_markup:
                payload["reply_markup"] = reply_markup

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        logger.warning(f"Telegram API error: {result.get('description')}")
                        status = "error"
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            status = "error"
        finally:
            latency = time.time() - start_time
            record_api_call("telegram", "sendMessage", status)
            record_api_latency("telegram", "sendMessage", latency)

    async def _answer_callback_async(
        self,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> None:
        """Answer a callback query."""
        import aiohttp
        import time

        if not TELEGRAM_BOT_TOKEN:
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/answerCallbackQuery"
            payload = {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        logger.warning(
                            f"Telegram callback answer failed: {result.get('description')}"
                        )
                        status = "error"
        except Exception as e:
            logger.error(f"Error answering Telegram callback: {e}")
            status = "error"
        finally:
            latency = time.time() - start_time
            record_api_call("telegram", "answerCallbackQuery", status)
            record_api_latency("telegram", "answerCallbackQuery", latency)

    async def _answer_inline_query_async(
        self,
        inline_query_id: str,
        results: List[Dict[str, Any]],
    ) -> None:
        """Answer an inline query."""
        import aiohttp
        import time

        if not TELEGRAM_BOT_TOKEN:
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/answerInlineQuery"
            payload = {
                "inline_query_id": inline_query_id,
                "results": results,
                "cache_time": 10,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        logger.warning(
                            f"Telegram inline answer failed: {result.get('description')}"
                        )
                        status = "error"
        except Exception as e:
            logger.error(f"Error answering Telegram inline query: {e}")
            status = "error"
        finally:
            latency = time.time() - start_time
            record_api_call("telegram", "answerInlineQuery", status)
            record_api_latency("telegram", "answerInlineQuery", latency)

    async def _send_voice_summary(
        self,
        chat_id: int,
        topic: str,
        final_answer: Optional[str],
        consensus_reached: bool,
        confidence: float,
        rounds_used: int,
    ) -> None:
        """Send a voice summary of the debate result."""
        try:
            from .tts_helper import get_tts_helper

            helper = get_tts_helper()
            if not helper.is_available:
                logger.debug("TTS not available for voice summary")
                return

            result = await helper.synthesize_debate_result(
                task=topic,
                final_answer=final_answer,
                consensus_reached=consensus_reached,
                confidence=confidence,
                rounds_used=rounds_used,
            )

            if result:
                await self._send_voice_async(
                    chat_id,
                    result.audio_bytes,
                    result.duration_seconds,
                )
        except Exception as e:
            logger.warning(f"Failed to send voice summary: {e}")

    async def _send_voice_async(
        self,
        chat_id: int,
        audio_bytes: bytes,
        duration: float,
    ) -> None:
        """Send a voice message to Telegram chat."""
        import aiohttp
        import io

        if not TELEGRAM_BOT_TOKEN:
            logger.warning("Cannot send voice: TELEGRAM_BOT_TOKEN not configured")
            return

        try:
            url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/sendVoice"

            # Create form data with audio file
            data = aiohttp.FormData()
            data.add_field("chat_id", str(chat_id))
            data.add_field(
                "voice",
                io.BytesIO(audio_bytes),
                filename="voice.ogg",
                content_type="audio/ogg",
            )
            data.add_field("duration", str(int(duration)))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    result = await response.json()
                    if not result.get("ok"):
                        logger.warning(f"Telegram sendVoice failed: {result.get('description')}")
                    else:
                        logger.info(f"Voice message sent to chat {chat_id}")
        except Exception as e:
            logger.error(f"Error sending Telegram voice: {e}")


# Export handler factory
_telegram_handler: Optional["TelegramHandler"] = None


def get_telegram_handler(server_context: Optional[Dict] = None) -> "TelegramHandler":
    """Get or create the Telegram handler instance."""
    global _telegram_handler
    if _telegram_handler is None:
        if server_context is None:
            server_context = {}
        _telegram_handler = TelegramHandler(server_context)
    return _telegram_handler


__all__ = ["TelegramHandler", "get_telegram_handler"]
