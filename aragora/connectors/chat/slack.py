"""
Slack Chat Connector.

Implements ChatPlatformConnector for Slack using
Slack's Web API and Block Kit.

Environment Variables:
- SLACK_BOT_TOKEN: Bot OAuth token (xoxb-...)
- SLACK_SIGNING_SECRET: For webhook verification
- SLACK_WEBHOOK_URL: For incoming webhooks
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .base import ChatPlatformConnector
from .models import (
    BotCommand,
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
    FileAttachment,
    InteractionType,
    MessageButton,
    MessageType,
    SendMessageResponse,
    UserInteraction,
    VoiceMessage,
    WebhookEvent,
)

# Environment configuration
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Slack API
SLACK_API_BASE = "https://slack.com/api"


class SlackConnector(ChatPlatformConnector):
    """
    Slack connector using Slack Web API.

    Supports:
    - Sending messages with Block Kit
    - Slash commands
    - Interactive components (buttons, menus)
    - File uploads
    - Threaded conversations
    - Ephemeral messages
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        signing_secret: Optional[str] = None,
        webhook_url: Optional[str] = None,
        **config: Any,
    ):
        """
        Initialize Slack connector.

        Args:
            bot_token: Bot OAuth token (defaults to SLACK_BOT_TOKEN)
            signing_secret: Webhook signing secret
            webhook_url: Incoming webhook URL
            **config: Additional configuration
        """
        super().__init__(
            bot_token=bot_token or SLACK_BOT_TOKEN,
            signing_secret=signing_secret or SLACK_SIGNING_SECRET,
            webhook_url=webhook_url or SLACK_WEBHOOK_URL,
            **config,
        )

    @property
    def platform_name(self) -> str:
        return "slack"

    @property
    def platform_display_name(self) -> str:
        return "Slack"

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send message to Slack channel."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            payload: dict[str, Any] = {
                "channel": channel_id,
                "text": text,
            }

            if blocks:
                payload["blocks"] = blocks

            if thread_id:
                payload["thread_ts"] = thread_id

            # Optional: unfurl links/media
            if "unfurl_links" in kwargs:
                payload["unfurl_links"] = kwargs["unfurl_links"]
            if "unfurl_media" in kwargs:
                payload["unfurl_media"] = kwargs["unfurl_media"]

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SLACK_API_BASE}/chat.postMessage",
                    headers=self._get_headers(),
                    json=payload,
                )
                data = response.json()

                if data.get("ok"):
                    return SendMessageResponse(
                        success=True,
                        message_id=data.get("ts"),
                        channel_id=data.get("channel"),
                        timestamp=data.get("ts"),
                    )
                else:
                    return SendMessageResponse(
                        success=False,
                        error=data.get("error", "Unknown error"),
                    )

        except Exception as e:
            logger.error(f"Slack send_message error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Update a Slack message."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            payload: dict[str, Any] = {
                "channel": channel_id,
                "ts": message_id,
                "text": text,
            }

            if blocks:
                payload["blocks"] = blocks

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SLACK_API_BASE}/chat.update",
                    headers=self._get_headers(),
                    json=payload,
                )
                data = response.json()

                if data.get("ok"):
                    return SendMessageResponse(
                        success=True,
                        message_id=data.get("ts"),
                        channel_id=data.get("channel"),
                    )
                else:
                    return SendMessageResponse(
                        success=False,
                        error=data.get("error"),
                    )

        except Exception as e:
            logger.error(f"Slack update_message error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a Slack message."""
        if not HTTPX_AVAILABLE:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SLACK_API_BASE}/chat.delete",
                    headers=self._get_headers(),
                    json={
                        "channel": channel_id,
                        "ts": message_id,
                    },
                )
                data = response.json()
                return data.get("ok", False)

        except Exception as e:
            logger.error(f"Slack delete_message error: {e}")
            return False

    async def send_ephemeral(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send ephemeral message visible only to one user."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            payload: dict[str, Any] = {
                "channel": channel_id,
                "user": user_id,
                "text": text,
            }

            if blocks:
                payload["blocks"] = blocks

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SLACK_API_BASE}/chat.postEphemeral",
                    headers=self._get_headers(),
                    json=payload,
                )
                data = response.json()

                return SendMessageResponse(
                    success=data.get("ok", False),
                    error=data.get("error"),
                )

        except Exception as e:
            logger.error(f"Slack send_ephemeral error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict]] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Slack slash command."""
        # Use response_url for async response
        if command.response_url:
            return await self._send_to_response_url(
                command.response_url,
                text,
                blocks,
                response_type="ephemeral" if ephemeral else "in_channel",
            )

        # Fallback to regular message
        if command.channel and command.user:
            if ephemeral:
                return await self.send_ephemeral(
                    command.channel.id,
                    command.user.id,
                    text,
                    blocks,
                )
            else:
                return await self.send_message(
                    command.channel.id,
                    text,
                    blocks,
                )

        return SendMessageResponse(success=False, error="No response target")

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Slack interaction."""
        if interaction.response_url:
            return await self._send_to_response_url(
                interaction.response_url,
                text,
                blocks,
                replace_original=replace_original,
            )

        if interaction.channel and interaction.message_id and replace_original:
            return await self.update_message(
                interaction.channel.id,
                interaction.message_id,
                text,
                blocks,
            )

        if interaction.channel:
            return await self.send_message(
                interaction.channel.id,
                text,
                blocks,
            )

        return SendMessageResponse(success=False, error="No response target")

    async def _send_to_response_url(
        self,
        response_url: str,
        text: str,
        blocks: Optional[list[dict]] = None,
        response_type: str = "ephemeral",
        replace_original: bool = False,
    ) -> SendMessageResponse:
        """Send response to Slack response_url."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            payload: dict[str, Any] = {
                "text": text,
                "response_type": response_type,
            }

            if blocks:
                payload["blocks"] = blocks

            if replace_original:
                payload["replace_original"] = True

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    response_url,
                    json=payload,
                )
                return SendMessageResponse(
                    success=response.status_code == 200,
                )

        except Exception as e:
            logger.error(f"Slack response_url error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def upload_file(
        self,
        channel_id: str,
        content: bytes,
        filename: str,
        content_type: str = "application/octet-stream",
        title: Optional[str] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FileAttachment:
        """Upload file to Slack."""
        if not HTTPX_AVAILABLE:
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
            )

        try:
            async with httpx.AsyncClient() as client:
                # Use files.upload API
                files = {"file": (filename, content, content_type)}
                data: dict[str, Any] = {
                    "channels": channel_id,
                    "filename": filename,
                }

                if title:
                    data["title"] = title

                if thread_id:
                    data["thread_ts"] = thread_id

                response = await client.post(
                    f"{SLACK_API_BASE}/files.upload",
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                    data=data,
                    files=files,
                )
                result = response.json()

                if result.get("ok"):
                    file_data = result.get("file", {})
                    return FileAttachment(
                        id=file_data.get("id", ""),
                        filename=file_data.get("name", filename),
                        content_type=file_data.get("mimetype", content_type),
                        size=file_data.get("size", len(content)),
                        url=file_data.get("url_private"),
                    )

                return FileAttachment(
                    id="",
                    filename=filename,
                    content_type=content_type,
                    size=len(content),
                )

        except Exception as e:
            logger.error(f"Slack upload_file error: {e}")
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
            )

    async def download_file(
        self,
        file_id: str,
        **kwargs: Any,
    ) -> FileAttachment:
        """Download file from Slack."""
        if not HTTPX_AVAILABLE:
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
            )

        try:
            # Get file info first
            async with httpx.AsyncClient() as client:
                info_response = await client.get(
                    f"{SLACK_API_BASE}/files.info",
                    headers=self._get_headers(),
                    params={"file": file_id},
                )
                info = info_response.json()

                if not info.get("ok"):
                    return FileAttachment(
                        id=file_id,
                        filename="",
                        content_type="application/octet-stream",
                        size=0,
                    )

                file_data = info.get("file", {})
                url = file_data.get("url_private_download") or file_data.get("url_private")

                if not url:
                    return FileAttachment(
                        id=file_id,
                        filename=file_data.get("name", ""),
                        content_type=file_data.get("mimetype", "application/octet-stream"),
                        size=file_data.get("size", 0),
                    )

                # Download the file
                download_response = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {self.bot_token}"},
                )

                return FileAttachment(
                    id=file_id,
                    filename=file_data.get("name", ""),
                    content_type=file_data.get("mimetype", "application/octet-stream"),
                    size=len(download_response.content),
                    url=url,
                    content=download_response.content,
                )

        except Exception as e:
            logger.error(f"Slack download_file error: {e}")
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
            )

    def format_blocks(
        self,
        title: Optional[str] = None,
        body: Optional[str] = None,
        fields: Optional[list[tuple[str, str]]] = None,
        actions: Optional[list[MessageButton]] = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Format content as Slack Block Kit blocks."""
        blocks: list[dict] = []

        if title:
            blocks.append(
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": title,
                        "emoji": True,
                    },
                }
            )

        if body:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": body,
                    },
                }
            )

        if fields:
            field_elements = []
            for label, value in fields:
                field_elements.append(
                    {
                        "type": "mrkdwn",
                        "text": f"*{label}*\n{value}",
                    }
                )
            blocks.append(
                {
                    "type": "section",
                    "fields": field_elements,
                }
            )

        if actions:
            action_elements = [
                self.format_button(btn.text, btn.action_id, btn.value, btn.style, btn.url) for btn in actions
            ]
            blocks.append(
                {
                    "type": "actions",
                    "elements": action_elements,
                }
            )

        return blocks

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: str = "default",
        url: Optional[str] = None,
    ) -> dict:
        """Format a Slack button element."""
        if url:
            return {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": text,
                    "emoji": True,
                },
                "url": url,
            }

        button: dict[str, Any] = {
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": text,
                "emoji": True,
            },
            "action_id": action_id,
            "value": value or action_id,
        }

        if style == "primary":
            button["style"] = "primary"
        elif style == "danger":
            button["style"] = "danger"

        return button

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """Verify Slack webhook signature."""
        if not self.signing_secret:
            return True

        timestamp = headers.get("X-Slack-Request-Timestamp", "")
        signature = headers.get("X-Slack-Signature", "")

        if not timestamp or not signature:
            return False

        # Check timestamp to prevent replay attacks
        try:
            request_time = int(timestamp)
            if abs(time.time() - request_time) > 300:
                return False
        except ValueError:
            return False

        # Compute expected signature
        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected = (
            "v0="
            + hmac.new(
                self.signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(expected, signature)

    def parse_webhook_event(
        self,
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

    def _parse_slash_command(self, parsed: dict) -> WebhookEvent:
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

    def _parse_interaction_payload(self, payload: dict) -> WebhookEvent:
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
                    interaction_type=InteractionType.BUTTON_CLICK
                    if action.get("type") == "button"
                    else InteractionType.SELECT_MENU,
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

    def _parse_event_callback(self, payload: dict) -> WebhookEvent:
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

    # ==========================================================================
    # Evidence Collection
    # ==========================================================================

    async def get_channel_history(
        self,
        channel_id: str,
        limit: int = 100,
        oldest: Optional[str] = None,
        latest: Optional[str] = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Get message history from a Slack channel.

        Uses conversations.history API to retrieve messages.

        Args:
            channel_id: Channel ID to get history from
            limit: Maximum number of messages (max 1000)
            oldest: Oldest timestamp to include
            latest: Latest timestamp to include
            **kwargs: Additional API parameters

        Returns:
            List of ChatMessage objects
        """
        if not HTTPX_AVAILABLE:
            logger.error("httpx not available for Slack API")
            return []

        try:
            params: dict[str, Any] = {
                "channel": channel_id,
                "limit": min(limit, 1000),  # Slack API max
            }

            if oldest:
                params["oldest"] = oldest
            if latest:
                params["latest"] = latest

            # Include thread replies if requested
            if kwargs.get("include_all_metadata", True):
                params["include_all_metadata"] = True

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SLACK_API_BASE}/conversations.history",
                    headers=self._get_headers(),
                    params=params,
                )
                data = response.json()

                if not data.get("ok"):
                    logger.error(f"Slack API error: {data.get('error')}")
                    return []

                messages: list[ChatMessage] = []
                channel_info = await self.get_channel_info(channel_id)
                channel = channel_info or ChatChannel(
                    id=channel_id,
                    platform=self.platform_name,
                )

                for msg in data.get("messages", []):
                    # Skip bot messages if configured
                    if kwargs.get("skip_bots", True) and msg.get("bot_id"):
                        continue

                    user = ChatUser(
                        id=msg.get("user", msg.get("bot_id", "")),
                        platform=self.platform_name,
                        is_bot=bool(msg.get("bot_id")),
                    )

                    chat_msg = ChatMessage(
                        id=msg.get("ts", ""),
                        platform=self.platform_name,
                        channel=channel,
                        author=user,
                        content=msg.get("text", ""),
                        thread_id=msg.get("thread_ts"),
                        timestamp=datetime.fromtimestamp(float(msg.get("ts", "0").split(".")[0])),
                        metadata={
                            "reply_count": msg.get("reply_count", 0),
                            "reactions": msg.get("reactions", []),
                        },
                    )
                    messages.append(chat_msg)

                return messages

        except Exception as e:
            logger.error(f"Error getting Slack channel history: {e}")
            return []

    async def collect_evidence(
        self,
        channel_id: str,
        query: Optional[str] = None,
        limit: int = 100,
        include_threads: bool = True,
        min_relevance: float = 0.0,
        **kwargs: Any,
    ) -> list[ChatEvidence]:
        """
        Collect chat messages as evidence for debates.

        Retrieves messages from a Slack channel, filters by relevance,
        and converts to ChatEvidence format with provenance tracking.

        Args:
            channel_id: Slack channel ID
            query: Optional search query to filter messages
            limit: Maximum number of messages to retrieve
            include_threads: Whether to include threaded replies
            min_relevance: Minimum relevance score for inclusion (0-1)
            **kwargs: Additional options

        Returns:
            List of ChatEvidence objects with relevance scoring
        """
        # Get channel history
        messages = await self.get_channel_history(
            channel_id=channel_id,
            limit=limit,
            **kwargs,
        )

        if not messages:
            return []

        # Convert to evidence with relevance scoring
        evidence_list: list[ChatEvidence] = []

        for msg in messages:
            # Calculate relevance
            relevance = self._compute_message_relevance(msg, query)

            # Skip low-relevance messages
            if relevance < min_relevance:
                continue

            # Create evidence
            evidence = ChatEvidence.from_message(
                message=msg,
                query=query,
                relevance_score=relevance,
            )

            evidence_list.append(evidence)

        # Sort by relevance (highest first)
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)

        # Optionally fetch thread replies for high-relevance messages
        if include_threads:
            await self._enrich_with_threads(evidence_list, limit=5, **kwargs)

        logger.info(
            f"Collected {len(evidence_list)} evidence items from Slack channel {channel_id}"
        )

        return evidence_list

    async def _enrich_with_threads(
        self,
        evidence_list: list[ChatEvidence],
        limit: int = 5,
        **kwargs: Any,
    ) -> None:
        """Enrich evidence with thread reply information."""
        if not HTTPX_AVAILABLE:
            return

        for evidence in evidence_list[:limit]:
            # Only enrich if this is a thread root with replies
            reply_count = evidence.metadata.get("reply_count", 0)
            if not evidence.is_thread_root or reply_count == 0:
                continue

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{SLACK_API_BASE}/conversations.replies",
                        headers=self._get_headers(),
                        params={
                            "channel": evidence.channel_id,
                            "ts": evidence.source_id,
                            "limit": 10,
                        },
                    )
                    data = response.json()

                    if data.get("ok"):
                        replies = data.get("messages", [])[1:]  # Skip root
                        evidence.reply_count = len(replies)
                        evidence.metadata["thread_replies"] = [
                            {
                                "text": r.get("text", "")[:200],
                                "user": r.get("user", ""),
                                "ts": r.get("ts", ""),
                            }
                            for r in replies
                        ]

            except Exception as e:
                logger.debug(f"Error enriching thread: {e}")

    async def search_messages(
        self,
        query: str,
        channel_id: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[ChatEvidence]:
        """
        Search for messages across Slack workspace.

        Uses Slack's search.messages API (requires search:read scope).

        Args:
            query: Search query
            channel_id: Optional channel to restrict search
            limit: Maximum results to return
            **kwargs: Additional search parameters

        Returns:
            List of ChatEvidence from matching messages
        """
        if not HTTPX_AVAILABLE:
            return []

        try:
            search_query = query
            if channel_id:
                search_query = f"in:<#{channel_id}> {query}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{SLACK_API_BASE}/search.messages",
                    headers=self._get_headers(),
                    params={
                        "query": search_query,
                        "count": limit,
                        "sort": kwargs.get("sort", "score"),
                    },
                )
                data = response.json()

                if not data.get("ok"):
                    logger.error(f"Slack search error: {data.get('error')}")
                    return []

                matches = data.get("messages", {}).get("matches", [])
                evidence_list: list[ChatEvidence] = []

                for match in matches:
                    channel = ChatChannel(
                        id=match.get("channel", {}).get("id", ""),
                        platform=self.platform_name,
                        name=match.get("channel", {}).get("name"),
                    )

                    user = ChatUser(
                        id=match.get("user", ""),
                        platform=self.platform_name,
                        username=match.get("username"),
                    )

                    msg = ChatMessage(
                        id=match.get("ts", ""),
                        platform=self.platform_name,
                        channel=channel,
                        author=user,
                        content=match.get("text", ""),
                        timestamp=datetime.fromtimestamp(
                            float(match.get("ts", "0").split(".")[0])
                        ),
                        metadata={
                            "permalink": match.get("permalink"),
                            "score": match.get("score"),
                        },
                    )

                    evidence = ChatEvidence.from_message(
                        message=msg,
                        query=query,
                        relevance_score=match.get("score", 1.0) / 100,  # Normalize
                    )
                    evidence.metadata["permalink"] = match.get("permalink")

                    evidence_list.append(evidence)

                return evidence_list

        except Exception as e:
            logger.error(f"Slack search error: {e}")
            return []
