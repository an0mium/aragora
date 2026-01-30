"""
Slack event handling: webhook verification, event parsing, slash commands, interactions.

Contains event subscription and webhook handling logic extracted from SlackConnector.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

from aragora.connectors.chat.models import (
    BotCommand,
    ChatChannel,
    ChatMessage,
    ChatUser,
    InteractionType,
    UserInteraction,
    WebhookEvent,
)


class _SlackConnectorProtocol(Protocol):
    """Protocol for methods expected by SlackEventsMixin from the main connector."""

    signing_secret: str | None

    @property
    def platform_name(self) -> str: ...

    # Internal methods from the mixin itself
    def _parse_slash_command(self, parsed: dict[str, Any]) -> WebhookEvent: ...
    def _parse_interaction_payload(self, payload: dict[str, Any]) -> WebhookEvent: ...
    def _parse_event_callback(self, payload: dict[str, Any]) -> WebhookEvent: ...


class SlackEventsMixin:
    """Mixin providing webhook verification and event parsing for Slack."""

    def verify_webhook(
        self: _SlackConnectorProtocol,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """Verify Slack webhook signature.

        SECURITY: Fails closed in production if signing_secret is not configured.
        Uses centralized webhook_security module for consistent verification.
        """
        from aragora.connectors.chat.webhook_security import verify_slack_signature

        result = verify_slack_signature(
            timestamp=headers.get("X-Slack-Request-Timestamp", ""),
            body=body,
            signature=headers.get("X-Slack-Signature", ""),
            signing_secret=self.signing_secret or "",
        )
        return result.verified

    def parse_webhook_event(
        self: _SlackConnectorProtocol,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse Slack webhook payload into WebhookEvent."""
        content_type = headers.get("Content-Type", "")

        # Handle URL-encoded form data (slash commands, interactions)
        if "application/x-www-form-urlencoded" in content_type:
            from urllib.parse import parse_qs

            parsed = parse_qs(body.decode("utf-8"))

            # Check for payload field (interactions)
            if "payload" in parsed:
                payload = json.loads(parsed["payload"][0])
                return self._parse_interaction_payload(payload)

            # Slash command
            return self._parse_slash_command(parsed)

        # Handle JSON (events API)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return WebhookEvent(
                platform=self.platform_name,
                event_type="error",
                raw_payload={},
            )

        # URL verification challenge
        if payload.get("type") == "url_verification":
            return WebhookEvent(
                platform=self.platform_name,
                event_type="url_verification",
                raw_payload=payload,
                challenge=payload.get("challenge"),
            )

        # Event callback
        if payload.get("type") == "event_callback":
            return self._parse_event_callback(payload)

        return WebhookEvent(
            platform=self.platform_name,
            event_type=payload.get("type", "unknown"),
            raw_payload=payload,
        )

    def _parse_slash_command(self: _SlackConnectorProtocol, parsed: dict[str, Any]) -> WebhookEvent:
        """Parse slash command from form data."""

        def get_first(key: str, default: str = "") -> str:
            values = parsed.get(key, [default])
            return values[0] if values else default

        user = ChatUser(
            id=get_first("user_id"),
            platform=self.platform_name,
            username=get_first("user_name"),
        )

        channel = ChatChannel(
            id=get_first("channel_id"),
            platform=self.platform_name,
            name=get_first("channel_name"),
            team_id=get_first("team_id"),
        )

        command_name = get_first("command").lstrip("/")
        command_text = get_first("text")

        return WebhookEvent(
            platform=self.platform_name,
            event_type="slash_command",
            raw_payload=parsed,
            command=BotCommand(
                name=command_name,
                text=f"/{command_name} {command_text}".strip(),
                args=command_text.split() if command_text else [],
                user=user,
                channel=channel,
                platform=self.platform_name,
                response_url=get_first("response_url"),
                metadata={"trigger_id": get_first("trigger_id")},
            ),
        )

    def _parse_interaction_payload(
        self: _SlackConnectorProtocol, payload: dict[str, Any]
    ) -> WebhookEvent:
        """Parse interactive component payload."""
        interaction_type = payload.get("type", "")

        user_data = payload.get("user", {})
        user = ChatUser(
            id=user_data.get("id", ""),
            platform=self.platform_name,
            username=user_data.get("username"),
            display_name=user_data.get("name"),
        )

        channel_data = payload.get("channel", {})
        channel = ChatChannel(
            id=channel_data.get("id", ""),
            platform=self.platform_name,
            name=channel_data.get("name"),
            team_id=payload.get("team", {}).get("id"),
        )

        event = WebhookEvent(
            platform=self.platform_name,
            event_type=interaction_type,
            raw_payload=payload,
        )

        if interaction_type == "block_actions":
            actions = payload.get("actions", [])
            if actions:
                action = actions[0]
                event.interaction = UserInteraction(
                    id=payload.get("trigger_id", ""),
                    interaction_type=(
                        InteractionType.BUTTON_CLICK
                        if action.get("type") == "button"
                        else InteractionType.SELECT_MENU
                    ),
                    action_id=action.get("action_id", ""),
                    value=action.get("value"),
                    values=action.get("selected_options", []),
                    user=user,
                    channel=channel,
                    message_id=payload.get("message", {}).get("ts"),
                    platform=self.platform_name,
                    response_url=payload.get("response_url"),
                )

        elif interaction_type == "view_submission":
            event.interaction = UserInteraction(
                id=payload.get("trigger_id", ""),
                interaction_type=InteractionType.MODAL_SUBMIT,
                action_id=payload.get("view", {}).get("callback_id", ""),
                user=user,
                channel=channel,
                platform=self.platform_name,
                response_url=payload.get("response_url"),
                metadata={"view": payload.get("view", {})},
            )

        return event

    def _parse_event_callback(
        self: _SlackConnectorProtocol, payload: dict[str, Any]
    ) -> WebhookEvent:
        """Parse Events API callback."""
        event_data = payload.get("event", {})
        event_type = event_data.get("type", "")

        event = WebhookEvent(
            platform=self.platform_name,
            event_type=event_type,
            raw_payload=payload,
        )

        if event_type == "message" and not event_data.get("bot_id"):
            user = ChatUser(
                id=event_data.get("user", ""),
                platform=self.platform_name,
            )

            channel = ChatChannel(
                id=event_data.get("channel", ""),
                platform=self.platform_name,
                team_id=payload.get("team_id"),
            )

            event.message = ChatMessage(
                id=event_data.get("ts", ""),
                platform=self.platform_name,
                channel=channel,
                author=user,
                content=event_data.get("text", ""),
                thread_id=event_data.get("thread_ts"),
            )

        return event
