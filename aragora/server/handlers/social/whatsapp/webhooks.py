"""
WhatsApp webhook verification and incoming webhook processing.

Handles:
- GET webhook verification (Meta challenge/response)
- POST webhook signature verification (HMAC-SHA256)
- Incoming webhook event parsing and dispatch
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

from ...base import (
    HandlerResult,
    auto_error_response,
    error_response,
    json_response,
)
from ...utils.rate_limit import rate_limit
from . import config as _config
from ..telemetry import (
    record_error,
    record_webhook_latency,
    record_webhook_request,
)

logger = logging.getLogger(__name__)


def verify_signature(handler: Any) -> bool:
    """Verify WhatsApp webhook signature.

    Meta signs webhooks using HMAC-SHA256 with the app secret.
    Signature is in X-Hub-Signature-256 header.

    SECURITY: Fails closed in production if WHATSAPP_APP_SECRET is not configured.
    """
    if not _config.WHATSAPP_APP_SECRET:
        env = os.environ.get("ARAGORA_ENV", "production").lower()
        is_production = env not in ("development", "dev", "local", "test")
        if is_production:
            logger.error(
                "SECURITY: WHATSAPP_APP_SECRET not configured in production. "
                "Rejecting webhook to prevent signature bypass."
            )
            return False
        logger.warning(
            "WHATSAPP_APP_SECRET not set - skipping signature verification. "
            "This is only acceptable in development!"
        )
        return True

    try:
        signature = handler.headers.get("X-Hub-Signature-256", "")
        if not signature or not signature.startswith("sha256="):
            return False

        # Read body for verification
        content_length = int(handler.headers.get("Content-Length", 0))
        if content_length > 10 * 1024 * 1024:
            return False
        body = handler.rfile.read(content_length)

        # Compute expected signature
        expected_sig = hmac.new(
            _config.WHATSAPP_APP_SECRET.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()

        actual_sig = signature[7:]  # Remove "sha256=" prefix
        return hmac.compare_digest(expected_sig, actual_sig)

    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Error verifying WhatsApp signature: {e}")
        return False


def verify_webhook(query_params: dict[str, Any]) -> HandlerResult:
    """Handle Meta webhook verification request.

    Meta sends a GET request with:
    - hub.mode=subscribe
    - hub.verify_token=<your_verify_token>
    - hub.challenge=<challenge_string>

    Must respond with the challenge if verify_token matches.
    """
    mode = (
        query_params.get("hub.mode", [""])[0]
        if isinstance(query_params.get("hub.mode"), list)
        else query_params.get("hub.mode", "")
    )
    token = (
        query_params.get("hub.verify_token", [""])[0]
        if isinstance(query_params.get("hub.verify_token"), list)
        else query_params.get("hub.verify_token", "")
    )
    challenge = (
        query_params.get("hub.challenge", [""])[0]
        if isinstance(query_params.get("hub.challenge"), list)
        else query_params.get("hub.challenge", "")
    )

    logger.info(f"WhatsApp webhook verification: mode={mode}, token={(token or '')[:10]}...")

    if mode == "subscribe":
        if _config.WHATSAPP_VERIFY_TOKEN:
            if token and hmac.compare_digest(token, _config.WHATSAPP_VERIFY_TOKEN):
                logger.info("WhatsApp webhook verified successfully")
                # Return challenge as plain text using HandlerResult
                return HandlerResult(
                    status_code=200,
                    content_type="text/plain",
                    body=challenge.encode("utf-8") if isinstance(challenge, str) else challenge,
                    headers={},
                )
        else:
            # Verification disabled; accept only when no token is provided
            if not token:
                logger.info("WhatsApp webhook verification skipped (no token configured)")
                return HandlerResult(
                    status_code=200,
                    content_type="text/plain",
                    body=challenge.encode("utf-8") if isinstance(challenge, str) else challenge,
                    headers={},
                )

    logger.warning("WhatsApp webhook verification failed")
    return error_response("Forbidden", 403)


class WebhookProcessor:
    """Processes incoming WhatsApp webhook events."""

    def __init__(self, message_handler: Any) -> None:
        """Initialize with a message handler (WhatsAppHandler instance).

        Args:
            message_handler: Object with _process_messages method.
        """
        self._message_handler = message_handler

    @auto_error_response("handle whatsapp webhook")
    @rate_limit(requests_per_minute=100, limiter_name="whatsapp_webhook")
    def handle_webhook(self, handler: Any) -> HandlerResult:
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
            if content_length > 10 * 1024 * 1024:
                return error_response("Request body too large", 413)
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
                        self._message_handler._process_messages(value)

            # Always respond 200 OK to acknowledge receipt
            return json_response({"status": "ok"})

        except json.JSONDecodeError:
            logger.warning("Invalid JSON in WhatsApp webhook")
            status = "error"
            record_error("whatsapp", "json_parse")
            return json_response({"status": "ok"})
        except (ValueError, KeyError, TypeError, RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"Error handling WhatsApp webhook: {e}", exc_info=True)
            status = "error"
            record_error("whatsapp", "unknown")
            return json_response({"status": "ok"})
        finally:
            latency = time.time() - start_time
            record_webhook_request("whatsapp", status)
            record_webhook_latency("whatsapp", latency)
