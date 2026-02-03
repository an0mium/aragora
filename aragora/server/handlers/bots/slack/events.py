"""
Slack Events API Handler.

This module handles incoming Slack Events API webhooks including:
- URL verification challenges
- App mentions
- Message events
- App uninstall/token revocation events
"""

import json
import logging
from typing import Any

from aragora.audit.unified import audit_data
from aragora.server.errors import safe_error_message
from aragora.server.handlers.base import HandlerResult, error_response, json_response
from aragora.server.handlers.utils.rate_limit import rate_limit

from .constants import (
    MAX_TOPIC_LENGTH,
    PERM_SLACK_COMMANDS_EXECUTE,
    RBAC_AVAILABLE,
    AuthorizationContext,
    check_permission,
    _validate_slack_channel_id,
    _validate_slack_input,
    _validate_slack_team_id,
    _validate_slack_user_id,
)

logger = logging.getLogger(__name__)


@rate_limit(rpm=60)
async def handle_slack_events(request: Any) -> HandlerResult:
    """Handle Slack Events API webhook.

    RBAC: Events from Slack are processed based on event type:
    - url_verification: No auth required (Slack challenge)
    - app_mention: Requires slack.commands.execute for command processing
    - app_uninstalled/tokens_revoked: System events, no user auth
    """
    try:
        body = await request.body()
        data = json.loads(body)

        # URL verification challenge - no auth required
        if data.get("type") == "url_verification":
            challenge = data.get("challenge", "")
            # Validate challenge to prevent injection
            if len(challenge) > 500:
                return error_response("Invalid challenge length", 400)
            return json_response({"challenge": challenge})

        # Extract team_id for workspace authorization
        team_id = data.get("team_id", "")
        event = data.get("event", {})
        event_type = event.get("type")

        # Validate team_id format for non-system events
        if event_type not in ("app_uninstalled", "tokens_revoked"):
            if team_id:
                valid, error = _validate_slack_team_id(team_id)
                if not valid:
                    logger.warning("Invalid team_id in event: %s", error)
                    return error_response(error or "Invalid team ID", 400)

        if event_type == "app_mention":
            # Bot was mentioned - could trigger a debate
            text = event.get("text", "")
            channel = event.get("channel", "")
            user = event.get("user", "")

            # Validate inputs
            valid, error = _validate_slack_user_id(user)
            if not valid:
                logger.warning("Invalid user_id in app_mention: %s", error)
                return json_response({"ok": True})  # Silent fail for invalid user

            valid, error = _validate_slack_channel_id(channel)
            if not valid:
                logger.warning("Invalid channel_id in app_mention: %s", error)
                return json_response({"ok": True})

            # Validate text input
            valid, error = _validate_slack_input(text, "text", MAX_TOPIC_LENGTH, allow_empty=True)
            if not valid:
                logger.warning("Invalid text in app_mention: %s", error)
                return json_response(
                    {
                        "response_type": "ephemeral",
                        "text": f"Invalid input: {error}",
                    }
                )

            # RBAC check for command execution
            if RBAC_AVAILABLE and check_permission is not None and team_id:
                try:
                    context = None
                    if AuthorizationContext is not None:
                        context = AuthorizationContext(
                            user_id=f"slack:{user}",
                            workspace_id=team_id,
                            roles={"user"},
                        )
                    if context:
                        decision = check_permission(context, PERM_SLACK_COMMANDS_EXECUTE)
                        if not decision.allowed:
                            logger.warning(
                                "Permission denied for app_mention: user=%s, team=%s",
                                user,
                                team_id,
                            )
                            return json_response(
                                {
                                    "response_type": "ephemeral",
                                    "text": "You do not have permission to execute commands.",
                                }
                            )
                except Exception as e:
                    logger.debug("RBAC check failed for app_mention: %s", e)

            logger.info(f"Slack mention from {user} in {channel}: {text[:100]}")

            # Parse command from mention
            # Format: @aragora ask "question" or @aragora status
            return json_response(
                {
                    "response_type": "in_channel",
                    "text": "Received your request. Processing...",
                }
            )

        elif event_type == "message":
            # Direct message or channel message
            pass

        elif event_type == "app_uninstalled":
            # App was uninstalled from workspace - clean up tokens
            # System event - no user permission check needed
            team_id = data.get("team_id") or event.get("team_id")
            if team_id:
                # Validate team_id even for uninstall events
                valid, _ = _validate_slack_team_id(team_id)
                if not valid:
                    logger.warning("Invalid team_id in app_uninstalled event")
                    return json_response({"ok": True})

                try:
                    from aragora.storage.slack_workspace_store import get_slack_workspace_store

                    store = get_slack_workspace_store()
                    store.revoke_token(team_id)
                    logger.info(f"Slack app uninstalled from workspace {team_id}")

                    audit_data(
                        user_id="system",
                        resource_type="slack_workspace",
                        resource_id=team_id,
                        action="uninstall",
                        platform="slack",
                    )
                except Exception as e:
                    logger.error(f"Failed to handle app_uninstalled for {team_id}: {e}")

            return json_response({"ok": True})

        elif event_type == "tokens_revoked":
            # Tokens were revoked (e.g., user deauthorized) - also clean up
            # System event - no user permission check needed
            team_id = data.get("team_id") or event.get("team_id")
            if team_id:
                # Validate team_id
                valid, _ = _validate_slack_team_id(team_id)
                if not valid:
                    logger.warning("Invalid team_id in tokens_revoked event")
                    return json_response({"ok": True})

                try:
                    from aragora.storage.slack_workspace_store import get_slack_workspace_store

                    store = get_slack_workspace_store()
                    store.revoke_token(team_id)
                    logger.info(f"Slack tokens revoked for workspace {team_id}")
                except Exception as e:
                    logger.error(f"Failed to handle tokens_revoked for {team_id}: {e}")

            return json_response({"ok": True})

        return json_response({"ok": True})

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.error(f"Slack events handler error: {e}")
        return error_response(safe_error_message(e, "Slack event"), 500)


__all__ = [
    "handle_slack_events",
]
