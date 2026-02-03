"""
Telegram webhook verification and processing.

Handles webhook signature verification, webhook configuration,
status reporting, and incoming update dispatch.
"""

from __future__ import annotations

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
from ..telemetry import (
    record_error,
    record_message,
    record_webhook_latency,
    record_webhook_request,
)

logger = logging.getLogger(__name__)


class TelegramWebhooksMixin:
    """Mixin providing webhook verification and processing for Telegram."""

    def _verify_secret(self, handler: Any) -> bool:
        """Verify Telegram webhook secret token.

        Telegram supports a secret_token parameter in setWebhook that is sent
        in the X-Telegram-Bot-Api-Secret-Token header.

        SECURITY: Fails closed in production if TELEGRAM_WEBHOOK_SECRET is not configured.
        """
        from aragora.server.handlers.social import telegram as telegram_module

        if not telegram_module.TELEGRAM_WEBHOOK_SECRET:
            env = os.environ.get("ARAGORA_ENV", "development").lower()
            is_production = env not in ("development", "dev", "local", "test")
            if is_production:
                logger.error(
                    "SECURITY: TELEGRAM_WEBHOOK_SECRET not configured in production. "
                    "Rejecting webhook to prevent signature bypass."
                )
                return False
            logger.warning(
                "TELEGRAM_WEBHOOK_SECRET not set - skipping verification. "
                "This is only acceptable in development!"
            )
            return True

        try:
            secret_header = handler.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            return hmac.compare_digest(secret_header, telegram_module.TELEGRAM_WEBHOOK_SECRET)
        except Exception as e:
            logger.warning("Error verifying Telegram secret: %s", e)
            return False

    def _get_status(self, handler: Any = None) -> HandlerResult:
        """Get Telegram integration status.

        RBAC Permission Required: telegram:read

        Args:
            handler: HTTP request handler (unused but kept for consistency)

        Returns:
            JSON response with integration status
        """
        from aragora.server.handlers.social import telegram as telegram_module

        return json_response(
            {
                "enabled": bool(telegram_module.TELEGRAM_BOT_TOKEN),
                "bot_token_configured": bool(telegram_module.TELEGRAM_BOT_TOKEN),
                "webhook_secret_configured": bool(telegram_module.TELEGRAM_WEBHOOK_SECRET),
            }
        )

    @auto_error_response("set telegram webhook")
    def _set_webhook(self, handler: Any) -> HandlerResult:
        """Configure Telegram webhook URL."""
        from aragora.server.handlers.social import telegram as telegram_module

        if not telegram_module.TELEGRAM_BOT_TOKEN:
            return error_response("Telegram bot token not configured", 500)

        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            webhook_url = data.get("url")
            if not webhook_url:
                return error_response("webhook url is required", 400)

            # Set webhook via Telegram API
            telegram_module.create_tracked_task(
                self._set_webhook_async(webhook_url),
                name="telegram-set-webhook",
            )

            return json_response({"status": "webhook configuration queued"})

        except json.JSONDecodeError:
            return error_response("Invalid JSON body", 400)
        except Exception as e:
            logger.error("Failed to set webhook: %s", e)
            return error_response(f"Failed to set webhook: {str(e)[:100]}", 500)

    async def _set_webhook_async(self, webhook_url: str) -> None:
        """Set Telegram webhook via API."""
        from aragora.server.http_client_pool import get_http_pool
        from aragora.server.handlers.social import telegram as telegram_module

        try:
            url = (
                f"{telegram_module.TELEGRAM_API_BASE}"
                f"{telegram_module.TELEGRAM_BOT_TOKEN}/setWebhook"
            )
            payload = {"url": webhook_url}

            if telegram_module.TELEGRAM_WEBHOOK_SECRET:
                payload["secret_token"] = telegram_module.TELEGRAM_WEBHOOK_SECRET

            pool = get_http_pool()
            async with pool.get_session("telegram") as client:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=30,
                )
                result = response.json()
                if result.get("ok"):
                    logger.info("Telegram webhook set to: %s", webhook_url)
                else:
                    logger.error("Failed to set webhook: %s", result)
        except Exception as e:
            logger.error("Error setting Telegram webhook: %s", e)

    @auto_error_response("handle telegram webhook")
    @rate_limit(requests_per_minute=100, limiter_name="telegram_webhook")
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

            logger.debug("Telegram update received: %s", update.get("update_id"))

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
            logger.error("Error handling Telegram webhook: %s", e, exc_info=True)
            status = "error"
            record_error("telegram", "unknown")
            return json_response({"ok": True})
        finally:
            latency = time.time() - start_time
            record_webhook_request("telegram", status)
            record_webhook_latency("telegram", latency)
