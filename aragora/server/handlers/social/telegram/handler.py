"""
Main TelegramHandler class.

Composes all Telegram functionality from mixin classes:
- TelegramRBACMixin: Permission checking
- TelegramMessagesMixin: Telegram API communication
- TelegramWebhooksMixin: Webhook verification and dispatch
- TelegramCommandsMixin: Bot command handling
- TelegramCallbacksMixin: Callback queries, inline queries, message handling
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...base import (
    BaseHandler,
    HandlerResult,
    error_response,
)
from ._common import (
    PERM_TELEGRAM_ADMIN,
    TelegramRBACMixin,
)
from .callbacks import TelegramCallbacksMixin
from .commands import TelegramCommandsMixin
from .messages import TelegramMessagesMixin
from .webhooks import TelegramWebhooksMixin

logger = logging.getLogger(__name__)


class TelegramHandler(
    TelegramRBACMixin,
    TelegramMessagesMixin,
    TelegramWebhooksMixin,
    TelegramCommandsMixin,
    TelegramCallbacksMixin,
    BaseHandler,
):
    """Handler for Telegram Bot integration endpoints."""

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/integrations/telegram/webhook",
        "/api/v1/integrations/telegram/status",
        "/api/v1/integrations/telegram/set-webhook",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route Telegram requests to appropriate methods."""
        logger.debug("Telegram request: %s", path)

        if path == "/api/v1/integrations/telegram/status":
            return self._get_status(handler)

        if path == "/api/v1/integrations/telegram/set-webhook":
            if handler.command != "POST":
                return error_response("Method not allowed", 405)
            perm_error = self._check_permission(handler, PERM_TELEGRAM_ADMIN)
            if perm_error:
                return perm_error
            return self._set_webhook(handler)

        if path == "/api/v1/integrations/telegram/webhook":
            if handler.command != "POST":
                return error_response("Method not allowed", 405)

            if not self._verify_secret(handler):
                logger.warning("Telegram webhook secret verification failed")
                return error_response("Unauthorized", 401)

            return self._handle_webhook(handler)

        return error_response("Not found", 404)

    def handle_post(self, path: str, body: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle POST requests."""
        return self.handle(path, {}, handler)


# Export handler factory
_telegram_handler: TelegramHandler | None = None


def get_telegram_handler(server_context: dict[str, Any] | None = None) -> TelegramHandler:
    """Get or create the Telegram handler instance."""
    global _telegram_handler
    if _telegram_handler is None:
        if server_context is None:
            server_context = dict()
        _telegram_handler = TelegramHandler(server_context)
    return _telegram_handler
