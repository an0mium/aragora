"""
WhatsApp Business API integration endpoint handlers.

Endpoints:
- GET  /api/integrations/whatsapp/webhook - Webhook verification (Meta verification)
- POST /api/integrations/whatsapp/webhook - Handle incoming messages
- GET  /api/integrations/whatsapp/status  - Get integration status

Environment Variables:
- WHATSAPP_ACCESS_TOKEN - Required for sending messages (Meta Business token)
- WHATSAPP_PHONE_NUMBER_ID - Required for sending messages
- WHATSAPP_VERIFY_TOKEN - Required for webhook verification
- WHATSAPP_APP_SECRET - Optional for signature verification

Supported Messages:
- Text messages → Commands or debate topics
- Interactive replies → Vote buttons

Bot Commands (send as text):
- help - Show available commands
- debate <topic> - Start a multi-agent debate
- gauntlet <statement> - Run adversarial validation
- status - Get system status
- agents - List available agents
"""

from __future__ import annotations

import asyncio
import hashlib
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
TTS_VOICE_ENABLED = os.environ.get("WHATSAPP_TTS_ENABLED", "false").lower() == "true"

# Environment variables for WhatsApp integration
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "aragora_verify_token")
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")
WHATSAPP_API_BASE = "https://graph.facebook.com/v18.0"


class WhatsAppHandler(BaseHandler):
    """Handler for WhatsApp Business API integration endpoints."""

    ROUTES = [
        "/api/integrations/whatsapp/webhook",
        "/api/integrations/whatsapp/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route WhatsApp requests to appropriate methods."""
        logger.debug(f"WhatsApp request: {path} {handler.command}")

        if path == "/api/integrations/whatsapp/status":
            return self._get_status()

        if path == "/api/integrations/whatsapp/webhook":
            if handler.command == "GET":
                # Webhook verification from Meta
                return self._verify_webhook(query_params)
            elif handler.command == "POST":
                # Verify signature if app secret is configured
                if WHATSAPP_APP_SECRET and not self._verify_signature(handler):
                    logger.warning("WhatsApp signature verification failed")
                    return error_response("Unauthorized", 401)
                return self._handle_webhook(handler)

        return error_response("Not found", 404)

    def handle_post(self, path: str, body: Dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """Handle POST requests."""
        return self.handle(path, {}, handler)

    def _verify_signature(self, handler: Any) -> bool:
        """Verify WhatsApp webhook signature.

        Meta signs webhooks using HMAC-SHA256 with the app secret.
        Signature is in X-Hub-Signature-256 header.
        """
        if not WHATSAPP_APP_SECRET:
            return True

        try:
            signature = handler.headers.get("X-Hub-Signature-256", "")
            if not signature or not signature.startswith("sha256="):
                return False

            # Read body for verification
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            # Compute expected signature
            expected_sig = hmac.new(
                WHATSAPP_APP_SECRET.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()

            actual_sig = signature[7:]  # Remove "sha256=" prefix
            return hmac.compare_digest(expected_sig, actual_sig)

        except Exception as e:
            logger.warning(f"Error verifying WhatsApp signature: {e}")
            return False

    def _get_status(self) -> HandlerResult:
        """Get WhatsApp integration status."""
        return json_response(
            {
                "enabled": bool(WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
                "access_token_configured": bool(WHATSAPP_ACCESS_TOKEN),
                "phone_number_id_configured": bool(WHATSAPP_PHONE_NUMBER_ID),
                "verify_token_configured": bool(WHATSAPP_VERIFY_TOKEN),
                "app_secret_configured": bool(WHATSAPP_APP_SECRET),
            }
        )

    def _verify_webhook(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Handle Meta webhook verification request.

        Meta sends a GET request with:
        - hub.mode=subscribe
        - hub.verify_token=<your_verify_token>
        - hub.challenge=<challenge_string>

        Must respond with the challenge if verify_token matches.
        """
        mode = query_params.get("hub.mode", [""])[0] if isinstance(query_params.get("hub.mode"), list) else query_params.get("hub.mode", "")
        token = query_params.get("hub.verify_token", [""])[0] if isinstance(query_params.get("hub.verify_token"), list) else query_params.get("hub.verify_token", "")
        challenge = query_params.get("hub.challenge", [""])[0] if isinstance(query_params.get("hub.challenge"), list) else query_params.get("hub.challenge", "")

        logger.info(f"WhatsApp webhook verification: mode={mode}, token={token[:10]}...")

        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            logger.info("WhatsApp webhook verified successfully")
            # Return challenge as plain text
            return {
                "status": 200,
                "headers": {"Content-Type": "text/plain"},
                "body": challenge,
            }

        logger.warning("WhatsApp webhook verification failed")
        return error_response("Forbidden", 403)

    @auto_error_response("handle whatsapp webhook")
    @rate_limit(rpm=100, limiter_name="whatsapp_webhook")
    def _handle_webhook(self, handler: Any) -> HandlerResult:
        """Handle incoming WhatsApp webhook events.

        Webhook structure:
        {
          "object": "whatsapp_business_account",
          "entry": [{
            "id": "<WHATSAPP_BUSINESS_ACCOUNT_ID>",
            "changes": [{
              "value": {
                "messaging_product": "whatsapp",
                "metadata": {...},
                "contacts": [...],
                "messages": [...],
                "statuses": [...]
              },
              "field": "messages"
            }]
          }]
        }
        """
        import time

        start_time = time.time()
        status = "success"
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            logger.debug(f"WhatsApp webhook received: {data.get('object')}")

            if data.get("object") != "whatsapp_business_account":
                return json_response({"status": "ok"})

            # Process each entry
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        value = change.get("value", {})
                        self._process_messages(value)

            # Always respond 200 OK to acknowledge receipt
            return json_response({"status": "ok"})

        except json.JSONDecodeError:
            logger.warning("Invalid JSON in WhatsApp webhook")
            status = "error"
            record_error("whatsapp", "json_parse")
            return json_response({"status": "ok"})
        except Exception as e:
            logger.error(f"Error handling WhatsApp webhook: {e}", exc_info=True)
            status = "error"
            record_error("whatsapp", "unknown")
            return json_response({"status": "ok"})
        finally:
            latency = time.time() - start_time
            record_webhook_request("whatsapp", status)
            record_webhook_latency("whatsapp", latency)

    def _process_messages(self, value: Dict[str, Any]) -> None:
        """Process incoming messages from webhook."""
        messages = value.get("messages", [])
        contacts = value.get("contacts", [])

        # Build contact lookup
        contact_map = {c.get("wa_id"): c for c in contacts}

        for message in messages:
            msg_type = message.get("type")
            from_number = message.get("from")
            contact = contact_map.get(from_number, {})
            profile_name = contact.get("profile", {}).get("name", "User")

            if msg_type == "text":
                text = message.get("text", {}).get("body", "")
                record_message("whatsapp", "text")
                self._handle_text_message(from_number, profile_name, text)
            elif msg_type == "interactive":
                record_message("whatsapp", "interactive")
                self._handle_interactive_reply(from_number, profile_name, message)
            elif msg_type == "button":
                record_message("whatsapp", "button")
                # Quick reply button
                button_text = message.get("button", {}).get("text", "")
                self._handle_button_reply(from_number, profile_name, button_text, message)

    def _handle_text_message(
        self,
        from_number: str,
        profile_name: str,
        text: str,
    ) -> None:
        """Handle incoming text message."""
        text = text.strip()
        logger.info(f"WhatsApp message from {profile_name} ({from_number}): {text[:50]}...")

        # Emit webhook event for message received
        emit_message_received(
            platform="whatsapp",
            chat_id=from_number,
            user_id=from_number,
            username=profile_name,
            message_text=text,
            message_type="text",
        )

        # Parse commands (lowercase first word)
        lower_text = text.lower()

        if lower_text == "help":
            record_command("whatsapp", "help")
            emit_command_received("whatsapp", from_number, from_number, profile_name, "help")
            response = self._command_help()
        elif lower_text == "status":
            record_command("whatsapp", "status")
            emit_command_received("whatsapp", from_number, from_number, profile_name, "status")
            response = self._command_status()
        elif lower_text == "agents":
            record_command("whatsapp", "agents")
            emit_command_received("whatsapp", from_number, from_number, profile_name, "agents")
            response = self._command_agents()
        elif lower_text.startswith("debate "):
            record_command("whatsapp", "debate")
            topic = text[7:].strip()
            emit_command_received("whatsapp", from_number, from_number, profile_name, "debate", topic)
            self._command_debate(from_number, profile_name, topic)
            return
        elif lower_text.startswith("gauntlet "):
            record_command("whatsapp", "gauntlet")
            statement = text[9:].strip()
            emit_command_received("whatsapp", from_number, from_number, profile_name, "gauntlet", statement)
            self._command_gauntlet(from_number, profile_name, statement)
            return
        elif len(text) > 10:
            # Treat longer messages as potential topics
            response = (
                f"I received: \"{text[:50]}...\"\n\n"
                "To start a debate, type:\n"
                f"debate {text[:50]}"
            )
        else:
            response = "Type *help* to see available commands."

        create_tracked_task(
            self._send_text_message_async(from_number, response),
            name=f"whatsapp-reply-{from_number}",
        )

    def _command_help(self) -> str:
        """Return help message."""
        return (
            "*Aragora Commands*\n\n"
            "*help* - Show this help message\n"
            "*status* - Get system status\n"
            "*agents* - List available agents\n"
            "*debate <topic>* - Start a multi-agent debate\n"
            "*gauntlet <statement>* - Run adversarial stress-test\n\n"
            "*Examples:*\n"
            "debate Should AI be regulated?\n"
            "gauntlet We should migrate to microservices"
        )

    def _command_status(self) -> str:
        """Return status message."""
        try:
            from aragora.ranking.elo import EloSystem

            store = EloSystem()
            agents = store.get_all_ratings()
            return f"*Aragora Status*\n\nStatus: Online\nAgents: {len(agents)} registered"
        except Exception as e:
            logger.warning(f"Failed to get status: {e}")
            return "*Aragora Status*\n\nStatus: Online"

    def _command_agents(self) -> str:
        """Return agents list."""
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
                lines.append(f"{i + 1}. *{name}* - ELO: {elo:.0f} | Wins: {wins}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to list agents: {e}")
            return "Could not fetch agent list."

    def _command_debate(
        self,
        from_number: str,
        profile_name: str,
        topic: str,
    ) -> None:
        """Handle debate command."""
        topic = topic.strip().strip("\"'")

        if len(topic) < 10:
            create_tracked_task(
                self._send_text_message_async(
                    from_number,
                    "Topic is too short. Please provide more detail.",
                ),
                name=f"whatsapp-debate-short-{from_number}",
            )
            return

        if len(topic) > 500:
            create_tracked_task(
                self._send_text_message_async(
                    from_number,
                    "Topic is too long. Please limit to 500 characters.",
                ),
                name=f"whatsapp-debate-long-{from_number}",
            )
            return

        # Send acknowledgment
        create_tracked_task(
            self._send_text_message_async(
                from_number,
                f"*Starting debate on:*\n_{topic}_\n\nRequested by {profile_name}\nProcessing... (this may take a few minutes)",
            ),
            name=f"whatsapp-debate-ack-{from_number}",
        )

        # Run debate asynchronously
        create_tracked_task(
            self._run_debate_async(from_number, profile_name, topic),
            name=f"whatsapp-debate-{topic[:30]}",
        )

    async def _run_debate_async(
        self,
        from_number: str,
        profile_name: str,
        topic: str,
    ) -> None:
        """Run debate and send result."""
        record_debate_started("whatsapp")
        debate_id = None
        try:
            from aragora import Arena, DebateProtocol, Environment
            from aragora.agents import get_agents_by_names  # type: ignore[attr-defined]

            # Register debate origin for tracking
            try:
                from aragora.server.debate_origin import register_debate_origin
                import uuid

                debate_id = f"wa-{from_number[-8:]}-{uuid.uuid4().hex[:8]}"
                register_debate_origin(
                    debate_id=debate_id,
                    platform="whatsapp",
                    channel_id=from_number,
                    user_id=from_number,
                    metadata={"profile_name": profile_name, "topic": topic},
                )
                logger.debug(f"Registered WhatsApp debate origin: {debate_id}")
            except ImportError:
                logger.debug("Debate origin tracking not available")

            # Emit webhook event for debate started
            emit_debate_started(
                platform="whatsapp",
                chat_id=from_number,
                user_id=from_number,
                username=profile_name,
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
                await self._send_text_message_async(
                    from_number,
                    "Failed to start debate: No agents available",
                )
                record_debate_failed("whatsapp")
                return

            arena = Arena.from_env(env, agents, protocol)
            result = await arena.run()

            response = (
                f"*Debate Complete!*\n\n"
                f"*Topic:* {topic[:100]}{'...' if len(topic) > 100 else ''}\n\n"
                f"*Consensus:* {'Yes' if result.consensus_reached else 'No'}\n"
                f"*Confidence:* {result.confidence:.1%}\n"
                f"*Rounds:* {result.rounds_used}\n"
                f"*Agents:* {len(agents)}\n\n"
                f"*Conclusion:*\n{result.final_answer[:500] if result.final_answer else 'No conclusion'}{'...' if result.final_answer and len(result.final_answer) > 500 else ''}\n\n"
                f"_Debate ID: {result.id}_\n"
                f"_Requested by {profile_name}_"
            )

            # Send result with interactive buttons
            await self._send_interactive_buttons_async(
                from_number,
                response,
                [
                    {"id": f"vote_agree_{result.id}", "title": "Agree"},
                    {"id": f"vote_disagree_{result.id}", "title": "Disagree"},
                    {"id": f"details_{result.id}", "title": "View Details"},
                ],
                "Vote on this debate",
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
                    from_number,
                    topic,
                    result.final_answer,
                    result.consensus_reached,
                    result.confidence,
                    result.rounds_used,
                )

            # Emit webhook event for debate completed
            emit_debate_completed(
                platform="whatsapp",
                chat_id=from_number,
                debate_id=result.id,
                topic=topic,
                consensus_reached=result.consensus_reached,
                confidence=result.confidence,
                rounds_used=result.rounds_used,
                final_answer=result.final_answer,
            )

            # Record successful debate completion
            record_debate_completed("whatsapp", result.consensus_reached)

        except Exception as e:
            logger.error(f"WhatsApp debate failed: {e}", exc_info=True)
            record_debate_failed("whatsapp")
            await self._send_text_message_async(
                from_number,
                f"Debate failed: {str(e)[:100]}",
            )

    def _command_gauntlet(
        self,
        from_number: str,
        profile_name: str,
        statement: str,
    ) -> None:
        """Handle gauntlet command."""
        statement = statement.strip().strip("\"'")

        if len(statement) < 10:
            create_tracked_task(
                self._send_text_message_async(
                    from_number,
                    "Statement is too short. Please provide more detail.",
                ),
                name=f"whatsapp-gauntlet-short-{from_number}",
            )
            return

        if len(statement) > 1000:
            create_tracked_task(
                self._send_text_message_async(
                    from_number,
                    "Statement is too long. Please limit to 1000 characters.",
                ),
                name=f"whatsapp-gauntlet-long-{from_number}",
            )
            return

        # Send acknowledgment
        create_tracked_task(
            self._send_text_message_async(
                from_number,
                f"*Running Gauntlet stress-test on:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\nRequested by {profile_name}\nRunning adversarial validation...",
            ),
            name=f"whatsapp-gauntlet-ack-{from_number}",
        )

        # Run gauntlet asynchronously
        create_tracked_task(
            self._run_gauntlet_async(from_number, profile_name, statement),
            name=f"whatsapp-gauntlet-{statement[:30]}",
        )

    async def _run_gauntlet_async(
        self,
        from_number: str,
        profile_name: str,
        statement: str,
    ) -> None:
        """Run gauntlet and send result."""
        import aiohttp

        record_gauntlet_started("whatsapp")

        # Emit webhook event for gauntlet started
        emit_gauntlet_started(
            platform="whatsapp",
            chat_id=from_number,
            user_id=from_number,
            username=profile_name,
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
                            "source": "whatsapp",
                            "from_number": from_number,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    data = await resp.json()

                    if resp.status != 200:
                        await self._send_text_message_async(
                            from_number,
                            f"Gauntlet failed: {data.get('error', 'Unknown error')}",
                        )
                        record_gauntlet_failed("whatsapp")
                        return

                    run_id = data.get("run_id", "unknown")
                    score = data.get("score", 0)
                    passed = data.get("passed", False)
                    vulnerabilities = data.get("vulnerabilities", [])

                    response = (
                        f"*Gauntlet Results* {'PASSED' if passed else 'FAILED'}\n\n"
                        f"*Statement:*\n_{statement[:200]}{'...' if len(statement) > 200 else ''}_\n\n"
                        f"*Score:* {score:.1%}\n"
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

                    response += f"\n_Run ID: {run_id}_\n_Requested by {profile_name}_"

                    await self._send_text_message_async(from_number, response)

                    # Emit webhook event for gauntlet completed
                    emit_gauntlet_completed(
                        platform="whatsapp",
                        chat_id=from_number,
                        gauntlet_id=run_id,
                        statement=statement,
                        verdict="passed" if passed else "failed",
                        confidence=score,
                        challenges_passed=len([v for v in vulnerabilities if not v.get("critical", False)]),
                        challenges_total=len(vulnerabilities) + 1,
                    )

                    # Record successful gauntlet completion
                    record_gauntlet_completed("whatsapp", passed)

        except Exception as e:
            logger.error(f"WhatsApp gauntlet failed: {e}", exc_info=True)
            record_gauntlet_failed("whatsapp")
            await self._send_text_message_async(
                from_number,
                f"Gauntlet failed: {str(e)[:100]}",
            )

    def _handle_interactive_reply(
        self,
        from_number: str,
        profile_name: str,
        message: Dict[str, Any],
    ) -> None:
        """Handle interactive message reply (button clicks)."""
        interactive = message.get("interactive", {})
        reply_type = interactive.get("type")

        if reply_type == "button_reply":
            button = interactive.get("button_reply", {})
            button_id = button.get("id", "")
            self._process_button_click(from_number, profile_name, button_id)
        elif reply_type == "list_reply":
            list_reply = interactive.get("list_reply", {})
            item_id = list_reply.get("id", "")
            self._process_button_click(from_number, profile_name, item_id)

    def _handle_button_reply(
        self,
        from_number: str,
        profile_name: str,
        button_text: str,
        message: Dict[str, Any],
    ) -> None:
        """Handle quick reply button."""
        # Map button text to action
        lower_text = button_text.lower()
        if "agree" in lower_text:
            # Extract debate_id from context if available
            context = message.get("context", {})
            # For quick replies, we might not have the ID directly
            logger.info(f"Quick reply 'agree' from {profile_name}")
        elif "disagree" in lower_text:
            logger.info(f"Quick reply 'disagree' from {profile_name}")

    def _process_button_click(
        self,
        from_number: str,
        profile_name: str,
        button_id: str,
    ) -> None:
        """Process button click by ID."""
        logger.info(f"Button click from {profile_name}: {button_id}")

        if button_id.startswith("vote_agree_"):
            debate_id = button_id[11:]
            self._record_vote(from_number, profile_name, debate_id, "agree")
        elif button_id.startswith("vote_disagree_"):
            debate_id = button_id[14:]
            self._record_vote(from_number, profile_name, debate_id, "disagree")
        elif button_id.startswith("details_"):
            debate_id = button_id[8:]
            self._send_debate_details(from_number, debate_id)

    def _record_vote(
        self,
        from_number: str,
        profile_name: str,
        debate_id: str,
        vote_option: str,
    ) -> None:
        """Record a vote."""
        logger.info(f"Vote received: {debate_id} -> {vote_option} from {profile_name}")

        # Emit webhook event for vote received
        emit_vote_received(
            platform="whatsapp",
            chat_id=from_number,
            user_id=from_number,
            username=profile_name,
            debate_id=debate_id,
            vote=vote_option,
        )

        # Record vote metrics
        record_vote("whatsapp", vote_option)

        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db and hasattr(db, "record_vote"):
                db.record_vote(
                    debate_id=debate_id,
                    voter_id=f"whatsapp:{from_number}",
                    vote=vote_option,
                    source="whatsapp",
                )
        except Exception as e:
            logger.warning(f"Failed to record vote: {e}")

        emoji = "+" if vote_option == "agree" else "-"
        create_tracked_task(
            self._send_text_message_async(
                from_number,
                f"{emoji} Your vote for '{vote_option}' has been recorded!",
            ),
            name=f"whatsapp-vote-ack-{from_number}",
        )

    def _send_debate_details(self, from_number: str, debate_id: str) -> None:
        """Send debate details."""
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
                self._send_text_message_async(
                    from_number,
                    f"Debate {debate_id} not found",
                ),
                name=f"whatsapp-details-notfound-{from_number}",
            )
            return

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
            f"*ID:* {debate_id}\n"
            f"*Consensus:* {'Yes' if consensus else 'No'}\n"
            f"*Confidence:* {confidence:.1%}\n"
            f"*Rounds:* {rounds_used}\n"
            f"*Agents:* {agent_list}\n\n"
            f"*Conclusion:*\n{final_answer[:500] if final_answer else 'No conclusion'}{'...' if final_answer and len(final_answer) > 500 else ''}"
        )

        create_tracked_task(
            self._send_text_message_async(from_number, response),
            name=f"whatsapp-details-{from_number}",
        )

    async def _send_text_message_async(
        self,
        to_number: str,
        text: str,
    ) -> None:
        """Send a text message via WhatsApp Cloud API."""
        import aiohttp
        import time

        if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            logger.warning("Cannot send message: WhatsApp not configured")
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_number,
                "type": "text",
                "text": {"preview_url": False, "body": text},
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.warning(f"WhatsApp API error: {result}")
                        status = "error"
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
            status = "error"
        finally:
            latency = time.time() - start_time
            record_api_call("whatsapp", "sendMessage", status)
            record_api_latency("whatsapp", "sendMessage", latency)

    async def _send_interactive_buttons_async(
        self,
        to_number: str,
        body_text: str,
        buttons: List[Dict[str, str]],
        header_text: Optional[str] = None,
    ) -> None:
        """Send an interactive buttons message.

        buttons: List of dicts with 'id' and 'title' keys (max 3 buttons)
        """
        import aiohttp
        import time

        if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            logger.warning("Cannot send message: WhatsApp not configured")
            return

        start_time = time.time()
        status = "success"
        try:
            url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"

            # WhatsApp allows max 3 buttons
            button_list = [
                {"type": "reply", "reply": {"id": b["id"], "title": b["title"][:20]}}
                for b in buttons[:3]
            ]

            interactive: Dict[str, Any] = {
                "type": "button",
                "body": {"text": body_text[:1024]},  # Max 1024 chars
                "action": {"buttons": button_list},
            }

            if header_text:
                interactive["header"] = {"type": "text", "text": header_text[:60]}

            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_number,
                "type": "interactive",
                "interactive": interactive,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.warning(f"WhatsApp API error: {result}")
                        status = "error"
                        # Fall back to plain text if interactive fails
                        await self._send_text_message_async(to_number, body_text)
        except Exception as e:
            logger.error(f"Error sending WhatsApp interactive message: {e}")
            status = "error"
            # Fall back to plain text
            await self._send_text_message_async(to_number, body_text)
        finally:
            latency = time.time() - start_time
            record_api_call("whatsapp", "sendInteractive", status)
            record_api_latency("whatsapp", "sendInteractive", latency)

    async def _send_voice_summary(
        self,
        to_number: str,
        topic: str,
        final_answer: Optional[str],
        consensus_reached: bool,
        confidence: float,
        rounds_used: int,
    ) -> None:
        """Send a voice summary of the debate result.

        Uses TTS to synthesize the result and sends as audio message.
        """
        try:
            from .tts_helper import get_tts_helper

            helper = get_tts_helper()
            if not helper.is_available:
                logger.debug("TTS not available for WhatsApp voice summary")
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
                    to_number,
                    result.audio_bytes,
                    result.format,
                )

        except ImportError:
            logger.debug("TTS helper not available")
        except Exception as e:
            logger.warning(f"Failed to send WhatsApp voice summary: {e}")

    async def _send_voice_async(
        self,
        to_number: str,
        audio_bytes: bytes,
        audio_format: str = "mp3",
    ) -> None:
        """Send an audio message via WhatsApp Cloud API.

        WhatsApp requires uploading media first, then sending with media ID.
        """
        import aiohttp

        if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            logger.warning("Cannot send voice: WhatsApp not configured")
            return

        try:
            # Step 1: Upload audio to WhatsApp Media API
            upload_url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/media"

            # Determine MIME type
            mime_types = {
                "mp3": "audio/mpeg",
                "ogg": "audio/ogg",
                "wav": "audio/wav",
                "m4a": "audio/mp4",
            }
            mime_type = mime_types.get(audio_format, "audio/mpeg")

            # Create form data for upload
            form_data = aiohttp.FormData()
            form_data.add_field(
                "file",
                audio_bytes,
                filename=f"voice.{audio_format}",
                content_type=mime_type,
            )
            form_data.add_field("messaging_product", "whatsapp")
            form_data.add_field("type", mime_type)

            async with aiohttp.ClientSession() as session:
                # Upload the audio file
                async with session.post(
                    upload_url,
                    data=form_data,
                    headers={
                        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                    },
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as upload_response:
                    upload_result = await upload_response.json()

                    if upload_response.status != 200:
                        logger.warning(f"WhatsApp media upload failed: {upload_result}")
                        return

                    media_id = upload_result.get("id")
                    if not media_id:
                        logger.warning("No media ID returned from upload")
                        return

                # Step 2: Send audio message with media ID
                send_url = f"{WHATSAPP_API_BASE}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to_number,
                    "type": "audio",
                    "audio": {"id": media_id},
                }

                async with session.post(
                    send_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as send_response:
                    send_result = await send_response.json()
                    if send_response.status != 200:
                        logger.warning(f"WhatsApp audio send failed: {send_result}")
                    else:
                        logger.info(f"WhatsApp voice message sent to {to_number}")

        except Exception as e:
            logger.error(f"Error sending WhatsApp voice message: {e}")


# Export handler factory
_whatsapp_handler: Optional["WhatsAppHandler"] = None


def get_whatsapp_handler(server_context: Optional[Dict] = None) -> "WhatsAppHandler":
    """Get or create the WhatsApp handler instance."""
    global _whatsapp_handler
    if _whatsapp_handler is None:
        if server_context is None:
            server_context = {}
        _whatsapp_handler = WhatsAppHandler(server_context)
    return _whatsapp_handler


__all__ = ["WhatsAppHandler", "get_whatsapp_handler"]
