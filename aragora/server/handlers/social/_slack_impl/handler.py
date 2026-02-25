"""
Slack integration handler - main SlackHandler class.

Composes all mixins (commands, events, interactive, blocks, messaging)
into the unified SlackHandler that routes incoming Slack requests.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from collections.abc import Awaitable
from urllib.parse import parse_qs


from . import config as _cfg
from .config import (
    BOTS_READ_PERMISSION,
    HandlerResult,
    SecureHandler,
    ForbiddenError,
    UnauthorizedError,
    error_response,
    json_response,
    get_slack_integration,
)

try:
    from aragora.server.handlers.base import handle_errors
except ImportError:

    def handle_errors(operation: str):  # type: ignore[misc]
        """No-op stub when handler base is unavailable."""

        def decorator(func):  # type: ignore[no-untyped-def]
            return func

        return decorator


from .commands import CommandsMixin
from .events import EventsMixin
from .interactive import InteractiveMixin

logger = logging.getLogger(__name__)


class SlackHandler(CommandsMixin, EventsMixin, InteractiveMixin, SecureHandler):
    """Handler for Slack integration endpoints.

    RBAC Protected:
    - bots.read - required for status endpoint

    Note: Webhook endpoints are authenticated via Slack's signature,
    not RBAC, since they are called by Slack servers directly.
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/integrations/slack/commands",
        "/api/v1/integrations/slack/interactive",
        "/api/v1/integrations/slack/events",
        "/api/v1/integrations/slack/status",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route Slack requests to appropriate methods."""
        logger.debug("Slack request: %s", path)

        if path == "/api/v1/integrations/slack/status":
            # RBAC: Require authentication and bots.read permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, BOTS_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning("Slack status access denied: %s", e)
                return error_response("Permission denied", 403)
            return self._get_status()

        # All other endpoints require POST
        if handler.command != "POST":
            return error_response("Method not allowed", 405)

        # Read and store body for signature verification and parsing
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            return error_response("Invalid Content-Length", 400)
        if content_length > 10 * 1024 * 1024:
            return error_response("Request body too large", 413)
        body = handler.rfile.read(content_length).decode("utf-8")

        # Extract team_id for multi-workspace support
        team_id = self._extract_team_id(body, path)
        workspace = _cfg.resolve_workspace(team_id) if team_id else None

        # Get signing secret (workspace-specific or fallback to env var)
        signing_secret = (
            workspace.signing_secret
            if workspace and workspace.signing_secret
            else _cfg.SLACK_SIGNING_SECRET
        )

        # Verify Slack signature for security - fail closed if secret missing in production
        if not signing_secret:
            env = os.environ.get("ARAGORA_ENV", "").lower()
            if env not in ("development", "dev", "local", "test"):
                logger.error("SECURITY: Slack signing secret not configured in production")
                return error_response("Webhook verification not configured", 503)
            logger.warning("Slack signing secret not configured - skipping in dev mode")
        elif not self._verify_signature(handler, body, signing_secret):
            logger.warning("Slack signature verification failed for team_id=%s", team_id)
            # Audit log signature failure (potential attack)
            audit = _cfg._get_audit_logger()
            if audit:
                ip_address = handler.client_address[0] if handler.client_address else ""
                user_agent = handler.headers.get("User-Agent", "")
                audit.log_signature_failure(
                    workspace_id=team_id or "",
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            return error_response("Invalid signature", 401)

        # Store workspace and body in handler for downstream methods
        handler._slack_workspace = workspace
        handler._slack_body = body
        handler._slack_team_id = team_id

        if path == "/api/v1/integrations/slack/commands":
            return self._handle_slash_command(handler)
        elif path == "/api/v1/integrations/slack/interactive":
            return self._handle_interactive(handler)
        elif path == "/api/v1/integrations/slack/events":
            return self._handle_events(handler)

        return error_response("Not found", 404)

    def _extract_team_id(self, body: str, path: str) -> str | None:
        """Extract team_id from request body based on endpoint type.

        Args:
            body: Raw request body
            path: Request path to determine parsing strategy

        Returns:
            team_id string or None
        """
        try:
            if path.endswith("/commands"):
                # Slash commands are form-encoded
                params = parse_qs(body)
                return params.get("team_id", [None])[0]
            elif path.endswith("/interactive"):
                # Interactive payloads are JSON in 'payload' field
                params = parse_qs(body)
                payload_str = params.get("payload", ["{}"])[0]
                payload = json.loads(payload_str)
                # Team info can be in 'team' or root
                team = payload.get("team", {})
                return team.get("id") or payload.get("team_id")
            elif path.endswith("/events"):
                # Events API sends JSON
                data = json.loads(body)
                # Team ID in event or root
                return data.get("team_id") or data.get("event", {}).get("team")
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
            logger.debug("Failed to extract team_id: %s", e)
        return None

    @handle_errors("slack creation")
    def handle_post(
        self, path: str, body: dict[str, Any], handler: Any
    ) -> Awaitable[HandlerResult | None]:
        """Handle POST requests.

        Returns an awaitable since handle() is async. Callers must await the result.
        """
        return self.handle(path, {}, handler)

    def _verify_signature(self, handler: Any, body: str, signing_secret: str) -> bool:
        """Verify Slack request signature.

        Uses centralized webhook verification for consistent security handling.
        See: https://api.slack.com/authentication/verifying-requests-from-slack

        Args:
            handler: HTTP request handler
            body: Pre-read request body
            signing_secret: Signing secret to use (workspace-specific or global)
        """
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        try:
            result = verify_slack_signature(
                timestamp=handler.headers.get("X-Slack-Request-Timestamp", ""),
                body=body,
                signature=handler.headers.get("X-Slack-Signature", ""),
                signing_secret=signing_secret or "",
            )
            if not result.verified and result.error:
                logger.warning("Slack signature verification failed: %s", result.error)
            return result.verified
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.exception("Unexpected signature verification error: %s", e)
            return False

    def _get_status(self) -> HandlerResult:
        """Get Slack integration status.

        Includes circuit breaker status for monitoring resilience.
        """
        from .messaging import get_slack_circuit_breaker

        integration = get_slack_integration()
        circuit_breaker = get_slack_circuit_breaker()

        return json_response(
            {
                "enabled": integration is not None,
                "signing_secret_configured": bool(_cfg.SLACK_SIGNING_SECRET),
                "bot_token_configured": bool(_cfg.SLACK_BOT_TOKEN),
                "webhook_configured": bool(_cfg.SLACK_WEBHOOK_URL),
                "circuit_breaker": circuit_breaker.get_status(),
            }
        )


# Export handler factory (lazy instantiation - server_context required)
_slack_handler: SlackHandler | None = None


def get_slack_handler(
    server_context: dict[str, Any] | dict[str, Any] | None = None,
) -> SlackHandler:
    """Get or create the Slack handler instance.

    Args:
        server_context: Server context dict (required for first call)

    Returns:
        SlackHandler instance
    """
    global _slack_handler
    if _slack_handler is None:
        if server_context is None:
            server_context = {}
        # Cast to ServerContext - the TypedDict accepts any dict with compatible keys
        _slack_handler = SlackHandler(server_context)
    return _slack_handler
