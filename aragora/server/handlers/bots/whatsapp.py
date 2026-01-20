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

    computed_sig = hmac.new(
        WHATSAPP_APP_SECRET.encode(),
        body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_sig, computed_sig)


class WhatsAppHandler(BaseHandler):
    """Handler for WhatsApp Cloud API webhook endpoints."""

    ROUTES = [
        "/api/bots/whatsapp/webhook",
        "/api/bots/whatsapp/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    @rate_limit(rpm=60)
    def handle(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Route WhatsApp GET requests."""
        if path == "/api/bots/whatsapp/status":
            return self._get_status()

        if path == "/api/bots/whatsapp/webhook":
            # Webhook verification challenge
            return self._handle_verification(query_params)

        return None

    @rate_limit(rpm=120)
    def handle_post(
        self, path: str, query_params: Dict[str, Any], handler: Any
    ) -> Optional[HandlerResult]:
        """Handle POST requests (webhook messages)."""
        if path == "/api/bots/whatsapp/webhook":
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
        phone_number_id = metadata.get("phone_number_id")

        contacts = value.get("contacts", [])
        messages = value.get("messages", [])

        for message in messages:
            msg_type = message.get("type")
            from_number = message.get("from")
            msg_id = message.get("id")
            timestamp = message.get("timestamp")

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
                    f"Unknown command: /{command}\n\nUse /help to see available commands."
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
                    logger.warning(f"WhatsApp send failed: {response.status_code} - {response.text}")

        except ImportError:
            logger.warning("httpx not available for WhatsApp messaging")
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")

    def _send_welcome(self, to_number: str) -> None:
        """Send welcome message."""
        self._send_message(
            to_number,
            "Welcome to Aragora - Omnivorous Multi Agent Decision Making Engine!\n\n"
            "I harness the collective intelligence of Claude, GPT, Gemini, Grok, and more "
            "to help you make better decisions through structured debate.\n\n"
            "Just send me a question and I'll start a multi-agent debate!\n\n"
            "Commands:\n"
            "/debate <question> - Start a debate\n"
            "/status - Check system status\n"
            "/help - Show help"
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
            "Should we use microservices or a monolith for our new project?"
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
            "Ready for debates!"
        )

    def _start_debate(self, to_number: str, contact_name: str, topic: str) -> None:
        """Start a debate on the given topic."""
        if not topic.strip():
            self._send_message(
                to_number,
                "Please provide a topic for the debate.\n\n"
                "Example: Should startups focus on growth or profitability first?"
            )
            return

        # TODO: Integrate with debate starter
        self._send_message(
            to_number,
            f"Starting debate on:\n\n{topic[:200]}\n\n"
            "I'll notify you when the AI agents reach consensus. "
            "This typically takes 2-5 minutes."
        )

        logger.info(f"Debate requested from WhatsApp {contact_name} ({to_number}): {topic[:100]}")
