"""
Shared constants, utilities, and RBAC helpers for the Telegram handler package.

This module contains:
- RBAC permission constants
- Environment variable configuration
- Async task helpers
- RBAC mixin for permission checking
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Coroutine

from ...base import (
    HandlerResult,
    error_response,
    json_response,
)

logger = logging.getLogger(__name__)

# ============================================================================
# RBAC Permission Constants for Telegram Integration
# ============================================================================

PERM_TELEGRAM_READ = "telegram:read"
PERM_TELEGRAM_MESSAGES_SEND = "telegram:messages:send"
PERM_TELEGRAM_DEBATES_CREATE = "telegram:debates:create"
PERM_TELEGRAM_DEBATES_READ = "telegram:debates:read"
PERM_TELEGRAM_GAUNTLET_RUN = "telegram:gauntlet:run"
PERM_TELEGRAM_VOTES_RECORD = "telegram:votes:record"
PERM_TELEGRAM_COMMANDS_EXECUTE = "telegram:commands:execute"
PERM_TELEGRAM_CALLBACKS_HANDLE = "telegram:callbacks:handle"
PERM_TELEGRAM_ADMIN = "telegram:admin"

# Environment variables for Telegram integration
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
TELEGRAM_API_BASE = "https://api.telegram.org/bot"

# Log warnings at module load time for missing secrets
if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN not configured - Telegram bot disabled")
if not TELEGRAM_WEBHOOK_SECRET:
    logger.warning("TELEGRAM_WEBHOOK_SECRET not configured - webhook verification disabled")


def _handle_task_exception(task: asyncio.Task[Any], task_name: str) -> None:
    """Handle exceptions from fire-and-forget async tasks."""
    if task.cancelled():
        logger.debug("Task %s was cancelled", task_name)
    elif task.exception():
        exc = task.exception()
        logger.error("Task %s failed with exception: %s", task_name, exc, exc_info=exc)


def create_tracked_task(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task[Any]:
    """Create an async task with exception logging."""
    try:
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(lambda t: _handle_task_exception(t, name))
        return task
    except RuntimeError:
        # No running loop (common in sync test contexts) - run inline.
        logger.debug("No running event loop; executing task inline: %s", name)
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            task = loop.create_task(coro, name=name)
            task.add_done_callback(lambda t: _handle_task_exception(t, name))
            loop.run_until_complete(task)
            return task
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# RBAC imports - optional dependency
check_permission: Callable[..., Any] | None
extract_user_from_request: Callable[..., Any] | None
AuthorizationContext: type[Any] | None

try:
    from aragora.rbac.checker import check_permission as _check_perm
    from aragora.rbac.models import AuthorizationContext as _AuthCtx
    from aragora.billing.auth import extract_user_from_request as _extract_user

    check_permission = _check_perm
    AuthorizationContext = _AuthCtx
    extract_user_from_request = _extract_user
    RBAC_AVAILABLE = True
except (ImportError, AttributeError):
    RBAC_AVAILABLE = False
    check_permission = None
    extract_user_from_request = None
    AuthorizationContext = None


def _tg():
    """Lazy import of the telegram package for patchable attribute access."""
    from aragora.server.handlers.social import telegram as telegram_module

    return telegram_module


class TelegramRBACMixin:
    """Mixin providing RBAC helper methods for Telegram handler."""

    def _get_auth_context(self, handler: Any) -> Any | None:
        """Extract authorization context from the HTTP request."""
        if not _tg().RBAC_AVAILABLE or extract_user_from_request is None:
            return None

        try:
            user_info = extract_user_from_request(handler)
            if not user_info:
                return None

            return AuthorizationContext(
                user_id=user_info.user_id or "anonymous",
                roles={user_info.role} if user_info.role else set(),
                org_id=user_info.org_id,
            )
        except Exception as e:
            logger.debug("Could not extract auth context: %s", e)
            return None

    def _get_telegram_user_context(
        self,
        user_id: int,
        username: str,
        chat_id: int,
    ) -> Any | None:
        """Build authorization context from Telegram user info."""
        if not _tg().RBAC_AVAILABLE or AuthorizationContext is None:
            return None

        try:
            return AuthorizationContext(
                user_id=f"telegram:{user_id}",
                roles={"telegram_user"},
                org_id=None,
            )
        except Exception as e:
            logger.debug("Could not create Telegram user context: %s", e)
            return None

    def _check_permission(self, handler: Any, permission_key: str) -> HandlerResult | None:
        """Check if current user has permission. Returns error response if denied."""
        if not _tg().RBAC_AVAILABLE or check_permission is None:
            return None

        context = self._get_auth_context(handler)
        if context is None:
            return None

        try:
            decision = check_permission(context, permission_key)
            if not decision.allowed:
                logger.warning("Permission denied: %s for user %s", permission_key, context.user_id)
                return error_response(f"Permission denied: {decision.reason}", 403)
        except Exception as e:
            logger.warning("RBAC check failed: %s", e)
            return None

        return None

    def _check_telegram_user_permission(
        self,
        user_id: int,
        username: str,
        chat_id: int,
        permission_key: str,
    ) -> bool:
        """Check if a Telegram user has permission for an action.

        Fails OPEN by default for Telegram users unless TELEGRAM_RBAC_ENABLED=true.
        """
        if not _tg().RBAC_AVAILABLE or check_permission is None:
            return True

        telegram_rbac_enabled = os.environ.get("TELEGRAM_RBAC_ENABLED", "false").lower() == "true"
        if not telegram_rbac_enabled:
            logger.debug(
                "Telegram RBAC not enabled (set TELEGRAM_RBAC_ENABLED=true to enforce), "
                "allowing %s for user %s",
                permission_key,
                user_id,
            )
            return True

        context = self._get_telegram_user_context(user_id, username, chat_id)
        if context is None:
            return True

        try:
            decision = check_permission(context, permission_key)
            if not decision.allowed:
                logger.warning(
                    "Telegram permission denied: %s for user %s (@%s) - reason: %s",
                    permission_key,
                    user_id,
                    username,
                    decision.reason,
                )
                return False
            return True
        except Exception as e:
            logger.warning("Telegram RBAC check failed: %s", e)
            return True

    def _deny_telegram_permission(
        self,
        chat_id: int,
        permission_key: str,
        action_description: str,
    ) -> HandlerResult:
        """Send permission denied message to Telegram chat and return OK response."""
        from aragora.server.handlers.social import telegram as telegram_module

        message = (
            f"Permission denied: You don't have permission to {action_description}.\n\n"
            f"Required permission: `{permission_key}`\n"
            "Contact your administrator if you need access."
        )
        telegram_module.create_tracked_task(
            self._send_message_async(chat_id, message, parse_mode="Markdown"),
            name=f"telegram-permission-denied-{chat_id}",
        )
        return json_response({"ok": True})
