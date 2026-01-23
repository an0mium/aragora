"""
WhatsApp Business webhook handler.

Handles WhatsApp Cloud API webhooks for bidirectional chat.

Endpoints:
- GET  /api/bots/whatsapp/webhook - Webhook verification
- POST /api/bots/whatsapp/webhook - Handle incoming messages
- GET  /api/bots/whatsapp/status - Get integration status

Environment Variables:
- WHATSAPP_VERIFY_TOKEN - Token for webhook verification
- WHATSAPP_ACCESS_TOKEN - Cloud API access token
- WHATSAPP_PHONE_NUMBER_ID - Business phone number ID
"""

from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Environment variables
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_APP_SECRET = os.environ.get("WHATSAPP_APP_SECRET", "")

# WhatsApp Cloud API
WHATSAPP_API_URL = "https://graph.facebook.com/v18.0"


def _verify_whatsapp_signature(signature: str, body: bytes) -> bool:
    """Verify WhatsApp webhook signature.

    WhatsApp uses HMAC-SHA256 with the app secret.
    See: https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests
    """
    if not WHATSAPP_APP_SECRET:
        logger.warning("WHATSAPP_APP_SECRET not configured, skipping signature verification")
        return True

    if not signature.startswith("sha256="):
        return False

    expected_sig = signature[7:]  # Remove "sha256=" prefix

    computed_sig = hmac.new(WHATSAPP_APP_SECRET.encode(), body, hashlib.sha256).hexdigest()

    return hmac.compare_digest(expected_sig, computed_sig)


class WhatsAppHandler(BaseHandler):
    """Handler for WhatsApp Cloud API webhook endpoints."""

    ROUTES = [
        "/api/v1/bots/whatsapp/webhook",
        "/api/v1/bots/whatsapp/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=60)
    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route WhatsApp GET requests."""
        if path == "/api/v1/bots/whatsapp/status":
            return self._get_status()

        if path == "/api/v1/bots/whatsapp/webhook":
            # Webhook verification challenge
            return self._handle_verification(query_params)

        return None

    @rate_limit(rpm=120)
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests (webhook messages)."""
        if path == "/api/v1/bots/whatsapp/webhook":
            return self._handle_webhook(handler)

        return None

    def _get_status(self) -> HandlerResult:
        """Get WhatsApp integration status."""
        return json_response(
            {
                "platform": "whatsapp",
                "enabled": bool(WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
                "access_token_configured": bool(WHATSAPP_ACCESS_TOKEN),
                "phone_number_configured": bool(WHATSAPP_PHONE_NUMBER_ID),
                "verify_token_configured": bool(WHATSAPP_VERIFY_TOKEN),
                "app_secret_configured": bool(WHATSAPP_APP_SECRET),
            }
        )

    def _handle_verification(self, query_params: Dict[str, Any]) -> HandlerResult:
        """Handle WhatsApp webhook verification challenge.

        WhatsApp sends a GET request with:
        - hub.mode=subscribe
        - hub.verify_token=<your token>
        - hub.challenge=<challenge string>

        We must respond with the challenge if the token matches.
        """
        mode = query_params.get("hub.mode", [""])[0]
        token = query_params.get("hub.verify_token", [""])[0]
        challenge = query_params.get("hub.challenge", [""])[0]

        if mode == "subscribe":
            if not WHATSAPP_VERIFY_TOKEN:
                logger.warning("WHATSAPP_VERIFY_TOKEN not configured")
                return error_response("Verify token not configured", 403)

            if hmac.compare_digest(token, WHATSAPP_VERIFY_TOKEN):
                logger.info("WhatsApp webhook verification successful")
                # Return challenge as plain text
                return HandlerResult(
                    status_code=200,
                    content_type="text/plain",
                    body=challenge.encode(),
                )
            else:
                logger.warning("WhatsApp verification token mismatch")
                return error_response("Invalid verify token", 403)

        return error_response("Invalid verification request", 400)

    def _handle_webhook(self, handler: Any) -> HandlerResult:
        """Handle WhatsApp webhook messages.

        WhatsApp sends POST requests with message updates.
        See: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks/payload-examples
        """
        try:
            # Verify signature if app secret is configured
            signature = handler.headers.get("X-Hub-Signature-256", "")
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length)

            if not _verify_whatsapp_signature(signature, body):
                logger.warning("WhatsApp signature verification failed")
                return error_response("Invalid signature", 401)

            # Parse webhook payload
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in WhatsApp webhook: {e}")
                return error_response("Invalid JSON", 400)

            # Process webhook entries
            for entry in payload.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        self._process_messages(change.get("value", {}))

            # Always return 200 to acknowledge
            return json_response({"status": "ok"})

        except Exception as e:
            logger.exception(f"Unexpected WhatsApp webhook error: {e}")
            # Return 200 to prevent retries
            return json_response({"status": "error", "message": str(e)[:100]})

    def _process_messages(self, value: Dict[str, Any]) -> None:
        """Process incoming WhatsApp messages."""
        metadata = value.get("metadata", {})
        metadata.get("phone_number_id")

        contacts = value.get("contacts", [])
        messages = value.get("messages", [])

        for message in messages:
            msg_type = message.get("type")
            from_number = message.get("from")
            msg_id = message.get("id")
            message.get("timestamp")

            # Find contact info
            contact_name = "Unknown"
            for contact in contacts:
                if contact.get("wa_id") == from_number:
                    contact_name = contact.get("profile", {}).get("name", from_number)
                    break

            logger.info(f"WhatsApp message from {contact_name} ({from_number}): type={msg_type}")

            if msg_type == "text":
                text = message.get("text", {}).get("body", "")
                self._handle_text_message(from_number, contact_name, text, msg_id)
            elif msg_type == "interactive":
                self._handle_interactive(from_number, message.get("interactive", {}))
            elif msg_type == "button":
                self._handle_button_reply(from_number, message.get("button", {}))
            else:
                logger.debug(f"Unhandled WhatsApp message type: {msg_type}")

    def _handle_text_message(
        self, from_number: str, contact_name: str, text: str, msg_id: str
    ) -> None:
        """Handle incoming text message."""
        text_lower = text.lower().strip()

        # Check for commands
        if text_lower.startswith("/"):
            parts = text[1:].split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "help":
                self._send_help(from_number)
            elif command == "debate":
                self._start_debate(from_number, contact_name, args)
            elif command == "status":
                self._send_status(from_number)
            else:
                self._send_message(
                    from_number,
                    f"Unknown command: /{command}\n\nUse /help to see available commands.",
                )
        elif text_lower in ("hi", "hello", "hey", "start"):
            self._send_welcome(from_number)
        else:
            # Treat as debate topic
            self._start_debate(from_number, contact_name, text)

    def _handle_interactive(self, from_number: str, interactive: Dict[str, Any]) -> None:
        """Handle interactive message response (list reply, button reply)."""
        int_type = interactive.get("type")

        if int_type == "list_reply":
            reply = interactive.get("list_reply", {})
            reply_id = reply.get("id", "")
            logger.info(f"WhatsApp list reply from {from_number}: {reply_id}")

        elif int_type == "button_reply":
            reply = interactive.get("button_reply", {})
            button_id = reply.get("id", "")
            logger.info(f"WhatsApp button reply from {from_number}: {button_id}")

    def _handle_button_reply(self, from_number: str, button: Dict[str, Any]) -> None:
        """Handle quick reply button response."""
        payload = button.get("payload", "")
        text = button.get("text", "")
        logger.info(f"WhatsApp button reply from {from_number}: {payload} ({text})")

    # Message sending

    def _send_message(self, to_number: str, text: str) -> None:
        """Send a text message via WhatsApp Cloud API."""
        if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
            logger.warning("Cannot send WhatsApp message: credentials not configured")
            return

        try:
            import httpx

            url = f"{WHATSAPP_API_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            headers = {
                "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            }
            data = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_number,
                "type": "text",
                "text": {"preview_url": False, "body": text},
            }

            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, headers=headers, json=data)
                if not response.is_success:
                    logger.warning(
                        f"WhatsApp send failed: {response.status_code} - {response.text}"
                    )

        except ImportError:
            logger.warning("httpx not available for WhatsApp messaging")
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")

    def _send_welcome(self, to_number: str) -> None:
        """Send welcome message."""
        self._send_message(
            to_number,
            "Welcome to Aragora - Control plane for multi-agent robust decisionmaking!\n\n"
            "I orchestrate 15+ AI models (Claude, GPT, Gemini, Grok and more) "
            "to debate and deliver defensible decisions.\n\n"
            "Just send me a question and I'll start a multi-agent robust decisionmaking!\n\n"
            "Commands:\n"
            "/debate <question> - Start a debate\n"
            "/status - Check system status\n"
            "/help - Show help",
        )

    def _send_help(self, to_number: str) -> None:
        """Send help message."""
        self._send_message(
            to_number,
            "Aragora Commands:\n\n"
            "/debate <question> - Start a multi-agent debate\n"
            "/status - Check Aragora system status\n"
            "/help - Show this message\n\n"
            "Or just send me any question to start a debate!\n\n"
            "Example:\n"
            "Should we use microservices or a monolith for our new project?",
        )

    def _send_status(self, to_number: str) -> None:
        """Send status message."""
        self._send_message(
            to_number,
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

    def _start_debate(self, to_number: str, contact_name: str, topic: str) -> None:
        """Start a debate on the given topic."""
        if not topic.strip():
            self._send_message(
                to_number,
                "Please provide a topic for the debate.\n\n"
                "Example: Should startups focus on growth or profitability first?",
            )
            return

        # Start debate via queue system
        debate_id = self._start_debate_async(to_number, contact_name, topic)

        self._send_message(
            to_number,
            f"Starting debate on:\n\n{topic[:200]}\n\n"
            "I'll notify you when the AI agents reach consensus. "
            f"Debate ID: {debate_id[:8]}...",
        )

        logger.info(f"Debate requested from WhatsApp {contact_name} ({to_number}): {topic[:100]}")

    def _start_debate_async(self, to_number: str, contact_name: str, topic: str) -> str:
        """Start a debate asynchronously via the DecisionRouter.

        Uses the unified DecisionRouter for:
        - Deduplication across channels
        - Response caching
        - RBAC enforcement
        - Consistent routing

        Falls back to queue system if DecisionRouter unavailable.
        """
        import uuid
        import asyncio

        debate_id = str(uuid.uuid4())

        # Register origin for result routing
        try:
            from aragora.server.debate_origin import register_debate_origin

            register_debate_origin(
                debate_id=debate_id,
                platform="whatsapp",
                channel_id=to_number,
                user_id=to_number,
                metadata={"topic": topic, "contact_name": contact_name},
            )
        except Exception as e:
            logger.warning(f"Failed to register debate origin: {e}")

        # Try DecisionRouter first (preferred)
        try:
            from aragora.core.decision import (
                DecisionRequest,
                DecisionType,
                InputSource,
                ResponseChannel,
                RequestContext,
                get_decision_router,
            )

            async def route_via_decision_router():
                request = DecisionRequest(
                    content=topic,
                    decision_type=DecisionType.DEBATE,
                    source=InputSource.WHATSAPP,
                    response_channels=[
                        ResponseChannel(
                            platform="whatsapp",
                            channel_id=to_number,
                            user_id=to_number,
                        )
                    ],
                    context=RequestContext(
                        user_id=f"whatsapp:{to_number}",
                        metadata={"contact_name": contact_name},
                    ),
                )

                # Route through DecisionRouter (handles origin registration, deduplication, caching)
                router = get_decision_router()
                result = await router.route(request)
                if result and result.debate_id:
                    logger.info(f"DecisionRouter started debate {result.debate_id} from WhatsApp")
                return result

            # Run async in background
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(route_via_decision_router())
                else:
                    asyncio.run(route_via_decision_router())
                return debate_id
            except RuntimeError:
                asyncio.run(route_via_decision_router())
                return debate_id

        except ImportError:
            logger.debug("DecisionRouter not available, falling back to queue system")
        except Exception as e:
            logger.error(f"DecisionRouter failed: {e}, falling back to queue system")

        # Fallback to queue system
        return self._start_debate_via_queue(to_number, contact_name, topic, debate_id)

    def _start_debate_via_queue(
        self, to_number: str, contact_name: str, topic: str, debate_id: str
    ) -> str:
        """Fallback to direct queue enqueue if DecisionRouter unavailable."""
        import asyncio

        try:
            from aragora.queue import create_debate_job

            job = create_debate_job(
                question=topic,
                agents=None,  # Use default agents
                rounds=3,
                consensus="majority",
                protocol="standard",
                user_id=f"whatsapp:{to_number}",
                webhook_url=None,  # Results routed via debate_origin system
            )

            # Fire and forget - enqueue the job
            async def enqueue_job():
                try:
                    from aragora.queue import create_redis_queue

                    queue = await create_redis_queue()
                    await queue.enqueue(job)
                    logger.info(f"WhatsApp debate job enqueued: {job.job_id}")
                except Exception as e:
                    logger.error(f"Failed to enqueue debate job: {e}")
                    self._send_message(
                        to_number, "Sorry, I couldn't start the debate. Please try again later."
                    )

            # Run async enqueue in background
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(enqueue_job())
                else:
                    asyncio.run(enqueue_job())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(enqueue_job())

            return job.job_id

        except ImportError:
            logger.warning("Queue system not available, using direct execution")
            # Fallback: run debate directly (blocking)
            return self._run_debate_direct(to_number, contact_name, topic, debate_id)
        except Exception as e:
            logger.error(f"Failed to start debate: {e}")
            return debate_id

    def _run_debate_direct(
        self, to_number: str, contact_name: str, topic: str, debate_id: str
    ) -> str:
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
                            to_number,
                            f"Debate Complete!\n\n"
                            f"Topic: {topic[:100]}\n\n"
                            f"Consensus: {result.final_answer[:500]}\n\n"
                            f"Confidence: {result.confidence:.0%}",
                        )
                    else:
                        self._send_message(
                            to_number,
                            f"Debate Complete!\n\n"
                            f"Topic: {topic[:100]}\n\n"
                            "No consensus was reached. The agents had differing views.",
                        )

                asyncio.run(execute())

            except Exception as e:
                logger.error(f"Direct debate execution failed: {e}")
                self._send_message(to_number, f"Debate failed: {str(e)[:100]}")

        # Run in background thread to not block webhook response
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        return debate_id
