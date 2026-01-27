"""Base mixin for bot integration handlers.

Provides shared patterns for bot webhook handlers including:
- Authenticated status endpoint handling
- Rate-limited webhook processing
- Consistent error handling and responses
- RBAC permission enforcement

This consolidates ~400 lines of duplicated code across 8 bot handlers:
- telegram.py, teams.py, slack.py, discord.py
- whatsapp.py, zoom.py, google_chat.py, email_webhook.py

Usage:
    from aragora.server.handlers.bots.base import BotHandlerMixin
    from aragora.server.handlers.secure import SecureHandler

    class MyBotHandler(BotHandlerMixin, SecureHandler):
        bot_platform = "mybot"

        async def handle(self, path, query_params, handler):
            if path.endswith("/status"):
                return await self.handle_status_request(handler)
            return None
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar

from aragora.server.handlers.base import HandlerResult, error_response, json_response
from aragora.server.handlers.utils.auth import UnauthorizedError, ForbiddenError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default RBAC permission for bot status endpoints
DEFAULT_BOTS_READ_PERMISSION = "bots.read"


class BotHandlerMixin:
    """Mixin providing shared patterns for bot integration handlers.

    Provides:
    - handle_status_request(): RBAC-protected status endpoint
    - handle_with_auth(): Generic auth wrapper for protected endpoints
    - Standard error response formatting

    Expected from SecureHandler (MRO):
    - get_auth_context(handler, require_auth) -> AuthorizationContext
    - check_permission(auth_context, permission) -> None
    """

    # Override in subclass to identify the bot platform
    bot_platform: str = "unknown"

    # Override to customize the permission required for status endpoint
    bots_read_permission: str = DEFAULT_BOTS_READ_PERMISSION

    async def handle_status_request(
        self,
        handler: Any,
        extra_status: Optional[Dict[str, Any]] = None,
    ) -> HandlerResult:
        """Handle RBAC-protected status endpoint request.

        Provides consistent auth checking and error handling across bot handlers.

        Args:
            handler: The HTTP request handler.
            extra_status: Additional status fields to include in response.

        Returns:
            HandlerResult with status JSON or error response.
        """
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)  # type: ignore[attr-defined]
            self.check_permission(auth_context, self.bots_read_permission)  # type: ignore[attr-defined]
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"{self.bot_platform.title()} status access denied: {e}")
            return error_response(str(e), 403)

        return self._build_status_response(extra_status)

    def _build_status_response(
        self, extra_status: Optional[Dict[str, Any]] = None
    ) -> HandlerResult:
        """Build the status response JSON.

        Override this method to customize status response per bot.

        Args:
            extra_status: Additional fields to include.

        Returns:
            HandlerResult with status JSON.
        """
        status = {
            "platform": self.bot_platform,
            "enabled": self._is_bot_enabled(),
        }
        if extra_status:
            status.update(extra_status)
        return json_response(status)

    def _is_bot_enabled(self) -> bool:
        """Check if this bot integration is enabled.

        Override in subclass to check environment variables or config.

        Returns:
            True if bot is configured and enabled.
        """
        return False

    async def handle_with_auth(
        self,
        handler: Any,
        permission: str,
        operation: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T | HandlerResult:
        """Execute an operation with RBAC authentication.

        Wraps the operation with standard auth checking and error handling.

        Args:
            handler: The HTTP request handler.
            permission: Required RBAC permission.
            operation: Async function to execute if authorized.
            *args: Positional arguments for operation.
            **kwargs: Keyword arguments for operation.

        Returns:
            Result of operation or error response.
        """
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)  # type: ignore[attr-defined]
            self.check_permission(auth_context, permission)  # type: ignore[attr-defined]
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(
                f"{self.bot_platform.title()} operation access denied "
                f"(permission={permission}): {e}"
            )
            return error_response(str(e), 403)

        return await operation(*args, auth_context=auth_context, **kwargs)

    def handle_rate_limit_exceeded(self, limit_info: Optional[str] = None) -> HandlerResult:
        """Return a rate limit exceeded response.

        Args:
            limit_info: Optional info about the rate limit that was exceeded.

        Returns:
            HandlerResult with 429 status and error message.
        """
        message = "Rate limit exceeded"
        if limit_info:
            message = f"{message}: {limit_info}"
        return error_response(message, 429)

    def handle_webhook_auth_failed(self, method: str = "unknown") -> HandlerResult:
        """Return an unauthorized response for webhook auth failure.

        Also logs the security event.

        Args:
            method: Authentication method that failed (e.g., "token", "signature").

        Returns:
            HandlerResult with 401 status.
        """
        logger.warning(f"{self.bot_platform.title()} webhook {method} verification failed")

        # Audit the failure if available
        try:
            from aragora.audit.unified import audit_security

            audit_security(
                event_type=f"{self.bot_platform}_webhook_auth_failed",
                actor_id="unknown",
                resource_type=f"{self.bot_platform}_webhook",
                resource_id=method,
            )
        except ImportError:
            pass  # Audit not available

        return error_response("Unauthorized", 401)


__all__ = [
    "BotHandlerMixin",
    "DEFAULT_BOTS_READ_PERMISSION",
]
