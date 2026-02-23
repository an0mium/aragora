"""
Slack Bot Handler - Main routing class.

This module contains the SlackHandler class which routes incoming
Slack webhook requests to the appropriate handler functions.
"""

import asyncio
import concurrent.futures
import logging
import os
from typing import Any

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
    handle_errors,
)
from aragora.server.handlers.bots.base import BotHandlerMixin
from aragora.server.handlers.secure import SecureHandler

from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed

from .commands import handle_slack_commands
from .constants import (
    PERM_SLACK_ADMIN,
    RBAC_AVAILABLE,
    check_permission,
)
from .events import handle_slack_events
from .interactions import handle_slack_interactions
from .signature import verify_slack_signature
from .state import _active_debates, get_slack_integration
from .user_management import (
    build_auth_context_from_slack,
    check_user_permission,
    check_workspace_authorized,
    get_org_from_team,
    get_user_roles_from_slack,
)

logger = logging.getLogger(__name__)


class SlackHandler(BotHandlerMixin, SecureHandler):
    """Handler for Slack bot integration endpoints.

    Uses BotHandlerMixin for shared auth/status patterns.
    Implements comprehensive RBAC protection for all operations.

    Security Model:
    - Slack signature verification (HMAC-SHA256) for request authentication
    - RBAC permission checks for authorization within verified requests
    - Input validation to prevent injection attacks
    - Workspace authorization to verify Slack workspace is allowed

    RBAC Permissions:
    - slack.commands.read: View status, help, and leaderboard
    - slack.commands.execute: Execute slash commands
    - slack.debates.create: Create new debates from Slack
    - slack.votes.record: Record votes in debates
    - slack.interactive.respond: Respond to interactive components
    - slack.admin: Full administrative access (bypasses other checks)

    Endpoints:
    - GET  /api/v1/bots/slack/status       - Get Slack integration status (no auth)
    - POST /api/v1/bots/slack/events       - Handle Slack Events API
    - POST /api/v1/bots/slack/interactions - Handle interactive components
    - POST /api/v1/bots/slack/commands     - Handle slash commands
    """

    bot_platform = "slack"

    ROUTES = [
        "/api/v1/bots/slack/status",
        "/api/v1/bots/slack/events",
        "/api/v1/bots/slack/interactions",
        "/api/v1/bots/slack/commands",
    ]

    def __init__(self, server_context: dict[str, Any]):
        """Initialize the Slack handler."""
        super().__init__(server_context)
        # Cache signing secret at init time (important for tests with patched env)
        self._signing_secret = os.environ.get("SLACK_SIGNING_SECRET")

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path.startswith("/api/v1/bots/slack/") or path.startswith(
            "/api/v1/integrations/slack/"
        )

    def _is_bot_enabled(self) -> bool:
        """Check if Slack bot is configured."""
        # Use package-level values so tests can patch slack.SLACK_BOT_TOKEN/SECRET.
        from aragora.server.handlers.bots import slack as slack_module

        return bool(slack_module.SLACK_BOT_TOKEN) or bool(slack_module.SLACK_SIGNING_SECRET)

    # =========================================================================
    # RBAC Helper Methods
    # =========================================================================

    def _build_auth_context_from_slack(
        self,
        team_id: str,
        user_id: str,
        channel_id: str | None = None,
    ) -> Any | None:
        """Build an AuthorizationContext from Slack request data."""
        return build_auth_context_from_slack(team_id, user_id, channel_id)

    def _get_org_from_team(self, team_id: str) -> str | None:
        """Get organization ID from Slack team/workspace ID."""
        return get_org_from_team(team_id)

    def _get_user_roles_from_slack(self, team_id: str, user_id: str) -> set[str]:
        """Get user roles based on Slack workspace membership."""
        return get_user_roles_from_slack(team_id, user_id)

    def _check_workspace_authorized(self, team_id: str) -> tuple[bool, str | None]:
        """Check if a Slack workspace is authorized to use Aragora."""
        return check_workspace_authorized(team_id)

    def _check_permission(
        self,
        team_id: str,
        user_id: str,
        permission_key: str,
        channel_id: str | None = None,
    ) -> HandlerResult | None:
        """Check if a Slack user has permission to perform an action."""
        return check_user_permission(team_id, user_id, permission_key, channel_id)

    def _check_permission_or_admin(
        self,
        team_id: str,
        user_id: str,
        permission_key: str,
        channel_id: str | None = None,
    ) -> HandlerResult | None:
        """Check if user has permission OR is admin."""
        if not RBAC_AVAILABLE or check_permission is None:
            if rbac_fail_closed():
                return error_response("Service unavailable: access control module not loaded", 503)
            return None

        context = self._build_auth_context_from_slack(team_id, user_id, channel_id)
        if context is None:
            return None

        try:
            # Check admin permission first
            admin_decision = check_permission(context, PERM_SLACK_ADMIN)
            if admin_decision.allowed:
                return None

            # Fall back to specific permission
            return self._check_permission(team_id, user_id, permission_key, channel_id)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            logger.debug("Admin check failed: %s", e)
            return self._check_permission(team_id, user_id, permission_key, channel_id)

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Handle requests for Slack endpoints.

        Handles:
        - GET /status endpoints
        - POST webhook endpoints (commands, events, interactions)
        """
        # Normalize paths: support both /api/v1/bots/slack and /api/integrations/slack
        normalized_path = path.replace("/api/integrations/slack", "/api/v1/bots/slack")
        normalized_path = normalized_path.replace(
            "/api/v1/integrations/slack", "/api/v1/bots/slack"
        )

        # Check HTTP method
        method = getattr(handler, "command", "GET")

        # Status endpoint - GET only
        if normalized_path in ["/api/v1/bots/slack/status", "/api/v1/bots/slack/status/"]:
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._get_status_sync(handler))
                    return future.result(timeout=5)
            except RuntimeError:
                # No running loop, create one
                return asyncio.run(self._get_status_sync(handler))
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
                logger.error("Error getting Slack status: %s", e)
                return error_response(safe_error_message(e, "Slack handler"), 500)

        # Webhook endpoints require POST
        webhook_paths = [
            "/api/v1/bots/slack/commands",
            "/api/v1/bots/slack/events",
            "/api/v1/bots/slack/interactions",
            "/api/v1/bots/slack/interactive",
        ]

        if normalized_path in webhook_paths or any(
            normalized_path.startswith(p) for p in webhook_paths
        ):
            if method != "POST":
                return error_response("Method not allowed. Use POST.", 405)

            # Verify Slack signature (use cached secret for test compatibility)
            signing_secret = getattr(self, "_signing_secret", "") or os.environ.get(
                "SLACK_SIGNING_SECRET", ""
            )
            if not signing_secret:
                logger.warning("SLACK_SIGNING_SECRET not configured - rejecting webhook request")
                return error_response("Slack signing secret not configured", 503)
            if not self._verify_signature(handler):
                return error_response("Invalid Slack signature", 401)

            # Handle the POST request
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.handle_post(normalized_path, query_params, handler),
                    )
                    return future.result(timeout=30)
            except RuntimeError:
                # No running loop, create one
                return asyncio.run(self.handle_post(normalized_path, query_params, handler))
            except (asyncio.TimeoutError, concurrent.futures.TimeoutError) as e:
                logger.error("Error handling Slack POST: %s", e)
                return error_response(safe_error_message(e, "Slack command"), 500)

        return None

    async def _get_status_sync(self, handler: Any) -> HandlerResult:
        """Get status using the mixin's handler."""
        extra_status = {
            "configured": self._is_bot_enabled(),
            "active_debates": len(_active_debates),
            "features": {
                "slash_commands": True,
                "events_api": True,
                "interactive_components": True,
                "block_kit": True,
            },
        }
        return await self.handle_status_request(handler, extra_status)

    @handle_errors("slack creation")
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle POST requests for Slack endpoints."""
        # Verify Slack signature for webhook endpoints
        if path in [
            "/api/v1/bots/slack/events",
            "/api/v1/bots/slack/interactions",
            "/api/v1/bots/slack/commands",
        ]:
            signing_secret = os.environ.get("SLACK_SIGNING_SECRET", "")
            if not signing_secret:
                logger.warning("SLACK_SIGNING_SECRET not configured - rejecting webhook request")
                return error_response("Slack signing secret not configured", 503)
            try:
                timestamp = handler.headers.get("X-Slack-Request-Timestamp", "")
                signature = handler.headers.get("X-Slack-Signature", "")
                body = handler.rfile.read(int(handler.headers.get("Content-Length", 0)))
                # Reset the file position for later reads
                handler.rfile.seek(0)

                if not verify_slack_signature(body, timestamp, signature, signing_secret):
                    return error_response("Invalid Slack signature", 401)
            except (ValueError, KeyError, AttributeError, OSError) as e:
                logger.warning("Slack signature verification error: %s", e)
                return error_response("Slack signature verification failed", 401)

        if path == "/api/v1/bots/slack/events":
            return await handle_slack_events(handler)

        if path == "/api/v1/bots/slack/interactions":
            return await handle_slack_interactions(handler)

        if path == "/api/v1/bots/slack/commands":
            return await handle_slack_commands(handler)

        return None

    def _verify_signature(self, handler: Any) -> bool:
        """Verify Slack request signature from handler.

        This is a convenience wrapper around verify_slack_signature for testing.
        """
        # Use cached signing secret (set at init time for test compatibility)
        signing_secret = getattr(self, "_signing_secret", "") or os.environ.get(
            "SLACK_SIGNING_SECRET", ""
        )
        if not signing_secret:
            env = os.environ.get("ARAGORA_ENV", "").lower()
            if env in ("development", "dev", "local", "test"):
                logger.warning("SLACK_SIGNING_SECRET not configured - skipping in dev mode")
                return True
            logger.error("SECURITY: SLACK_SIGNING_SECRET not configured in production")
            return False

        try:
            timestamp = handler.headers.get("X-Slack-Request-Timestamp", "")
            signature = handler.headers.get("X-Slack-Signature", "")

            # Try to get body from different sources
            if hasattr(handler, "_body"):
                body = handler._body
            elif hasattr(handler, "rfile"):
                content_length = int(handler.headers.get("Content-Length", 0))
                body = handler.rfile.read(content_length)
            else:
                body = b""

            return verify_slack_signature(body, timestamp, signature, signing_secret)
        except (ValueError, KeyError, AttributeError, OSError) as e:
            logger.warning("Signature verification error: %s", e)
            return False

    def _get_status(self) -> HandlerResult:
        """Get Slack integration status.

        Returns status information about the Slack integration.
        """
        integration = get_slack_integration()
        return json_response(
            {
                "enabled": self._is_bot_enabled(),
                "configured": integration is not None,
                "active_debates": len(_active_debates),
                "features": {
                    "slash_commands": True,
                    "events_api": True,
                    "interactive_components": True,
                    "block_kit": True,
                },
            }
        )

    def _command_help(self) -> HandlerResult:
        """Return help text for Slack commands.

        Returns an ephemeral message with available commands and examples.
        """
        help_text = """*Aragora Slack Commands*

*Available Commands:*
* `/aragora debate <topic>` - Start a debate on a topic
* `/aragora plan <topic>` - Debate + implementation plan
* `/aragora implement <topic>` - Debate + plan with context snapshot
* `/aragora status` - Show system status and active debates
* `/aragora help` - Show this help message
* `/aragora agents` - List available agents with ELO ratings
* `/aragora vote <debate_id> <agent>` - Vote for an agent in a debate

*Examples:*
* `/aragora debate Should AI be regulated?`
* `/aragora debate "What's the best programming language?"`
* `/aragora plan Draft a policy update for Q2`
* `/aragora status`

Need more help? Visit https://aragora.ai/docs/slack"""

        return json_response(
            {
                "response_type": "ephemeral",
                "text": help_text,
            }
        )

    def _command_status(self) -> HandlerResult:
        """Return system status for Slack.

        Returns status information formatted for Slack.
        """
        error_msg = None
        try:
            from aragora.ranking.elo import get_elo_store

            elo_store = get_elo_store()
            ratings = elo_store.get_all_ratings()
            agent_count = len(ratings)
        except (ImportError, AttributeError, RuntimeError) as e:
            agent_count = 0
            error_msg = type(e).__name__
        except (TypeError, ValueError, KeyError, OSError) as e:
            logger.exception("Unexpected error getting status: %s", e)
            agent_count = 0
            error_msg = "unexpected error"

        status_text = f"""*System Status: Online*

* Active debates: {len(_active_debates)}
* Registered agents: {agent_count}
* Integration: {"Configured" if self._is_bot_enabled() else "Not configured"}"""

        if error_msg:
            status_text += f"\n* Error: {error_msg}"

        return json_response(
            {
                "response_type": "ephemeral",
                "text": status_text,
            }
        )

    def _command_agents(self) -> HandlerResult:
        """Return list of available agents with ELO ratings.

        Returns an ephemeral message listing agents sorted by ELO.
        """
        try:
            from aragora.ranking.elo import get_elo_store

            elo_store = get_elo_store()
            ratings = elo_store.get_all_ratings()

            if not ratings:
                return json_response(
                    {
                        "response_type": "ephemeral",
                        "text": "No agents registered yet.",
                    }
                )

            # Sort by ELO rating descending
            sorted_ratings = sorted(ratings, key=lambda x: x.elo, reverse=True)

            agent_lines = [
                f"* {r.agent_name}: {r.elo:.0f} ELO"
                for r in sorted_ratings[:10]  # Top 10
            ]

            return json_response(
                {
                    "response_type": "ephemeral",
                    "text": "*Available Agents (by ELO):*\n" + "\n".join(agent_lines),
                }
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.error("Error getting agents: %s", e)
            return json_response(
                {
                    "response_type": "ephemeral",
                    "text": "Error fetching agents. Please try again later.",
                }
            )
        except (TypeError, ValueError, KeyError, OSError) as e:
            logger.exception("Unexpected error getting agents: %s", e)
            return json_response(
                {
                    "response_type": "ephemeral",
                    "text": "An unexpected error occurred. Please try again later.",
                }
            )

    def _slack_response(self, text: str, response_type: str = "in_channel") -> HandlerResult:
        """Create a Slack response with text.

        Args:
            text: Response text
            response_type: "ephemeral" (only visible to user) or "in_channel" (visible to all)
        """
        return json_response(
            {
                "response_type": response_type,
                "text": text,
            }
        )

    def _slack_blocks_response(
        self,
        blocks: list[dict[str, Any]],
        text: str = "",
        response_type: str = "in_channel",
    ) -> HandlerResult:
        """Create a Slack response with Block Kit blocks.

        Args:
            blocks: Block Kit blocks
            text: Fallback text for notifications
            response_type: "ephemeral" (only visible to user) or "in_channel" (visible to all)
        """
        response: dict[str, Any] = {
            "response_type": response_type,
            "blocks": blocks,
        }
        if text:
            response["text"] = text
        return json_response(response)


def register_slack_routes(router: Any) -> None:
    """Register Slack routes with the server router.

    Note: This function is deprecated in favor of using SlackHandler class
    with the unified handler registration system.
    """

    async def events_handler(request: Any) -> HandlerResult:
        return await handle_slack_events(request)

    async def interactions_handler(request: Any) -> HandlerResult:
        return await handle_slack_interactions(request)

    async def commands_handler(request: Any) -> HandlerResult:
        return await handle_slack_commands(request)

    # Register routes
    router.add_route("POST", "/api/bots/slack/events", events_handler)
    router.add_route("POST", "/api/bots/slack/interactions", interactions_handler)
    router.add_route("POST", "/api/bots/slack/commands", commands_handler)


__all__ = [
    "SlackHandler",
    "register_slack_routes",
]
