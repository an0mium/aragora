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

import json
import logging
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple, TypeVar

from aragora.server.handlers.base import HandlerResult, error_response, json_response
from aragora.server.handlers.utils.auth import ForbiddenError, UnauthorizedError

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

    # =========================================================================
    # Request body utilities - consolidates ~80 lines of duplicated code
    # =========================================================================

    def _read_request_body(self, handler: Any) -> bytes:
        """Read the request body from the handler.

        Handles Content-Length header parsing and body reading.

        Args:
            handler: The HTTP request handler with headers and rfile.

        Returns:
            The raw request body as bytes.
        """
        content_length = int(handler.headers.get("Content-Length", 0))
        if content_length <= 0:
            return b""
        return handler.rfile.read(content_length)

    def _parse_json_body(
        self, body: bytes, context: str = "webhook", allow_empty: bool = False
    ) -> Tuple[Optional[Dict[str, Any]], Optional[HandlerResult]]:
        """Parse JSON from request body with standardized error handling.

        Args:
            body: Raw request body bytes.
            context: Context string for error logging (e.g., "webhook", "event").
            allow_empty: If True, empty body returns ({}, None). If False, returns error.

        Returns:
            Tuple of (parsed_data, error_response).
            If parsing succeeds: (dict, None)
            If parsing fails: (None, HandlerResult with 400 error)
            If body is empty and allow_empty: ({}, None)
            If body is empty and not allow_empty: (None, HandlerResult with 400 error)
        """
        if not body:
            if allow_empty:
                return {}, None
            logger.error(f"Empty body in {self.bot_platform} {context}")
            return None, error_response("Empty request body", 400)

        try:
            return json.loads(body.decode("utf-8")), None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.bot_platform} {context}: {e}")
            return None, error_response("Invalid JSON", 400)

    def _handle_webhook_exception(
        self,
        exception: Exception,
        context: str = "webhook",
        return_200_on_error: bool = True,
    ) -> HandlerResult:
        """Handle webhook exceptions with standardized logging and responses.

        Many bot platforms require 200 responses even on error to prevent retries.
        This method provides consistent exception handling across all bot handlers.

        Args:
            exception: The caught exception.
            context: Context string for logging (e.g., "webhook", "event").
            return_200_on_error: If True, return 200 with error in body (prevents retries).
                               If False, return appropriate error status code.

        Returns:
            HandlerResult with appropriate status and error message.
        """
        error_msg = str(exception)[:100]

        if isinstance(exception, json.JSONDecodeError):
            logger.error(f"Invalid JSON in {self.bot_platform} {context}: {exception}")
            return error_response("Invalid JSON payload", 400)

        if isinstance(exception, (ValueError, KeyError, TypeError)):
            logger.warning(f"Data error in {self.bot_platform} {context}: {exception}")
            if return_200_on_error:
                return json_response({"ok": False, "error": error_msg})
            return error_response(f"Invalid data: {error_msg}", 400)

        if isinstance(exception, (ConnectionError, OSError, TimeoutError)):
            logger.error(f"Connection error in {self.bot_platform} {context}: {exception}")
            if return_200_on_error:
                return json_response({"ok": False, "error": "Connection error"})
            return error_response("Service temporarily unavailable", 503)

        # Unexpected exception
        logger.exception(f"Unexpected {self.bot_platform} {context} error: {exception}")
        if return_200_on_error:
            return json_response({"ok": False, "error": error_msg})
        return error_response(f"Internal error: {error_msg}", 500)

    def _audit_webhook_auth_failure(self, method: str, reason: Optional[str] = None) -> None:
        """Audit a webhook authentication failure.

        Args:
            method: Authentication method that failed (e.g., "signature", "token").
            reason: Optional additional reason for the failure.
        """
        try:
            from aragora.audit.unified import audit_security

            audit_security(
                event_type=f"{self.bot_platform}_webhook_auth_failed",
                actor_id="unknown",
                resource_type=f"{self.bot_platform}_webhook",
                resource_id=method,
                reason=reason,
            )
        except ImportError:
            pass  # Audit not available


__all__ = [
    "BotHandlerMixin",
    "DEFAULT_BOTS_READ_PERMISSION",
]
