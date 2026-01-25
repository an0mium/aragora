"""
Discord Chat Connector.

Implements ChatPlatformConnector for Discord using
Discord's REST API and Interactions.

Includes circuit breaker protection for all API calls to handle
rate limiting and service outages gracefully.

Environment Variables:
- DISCORD_BOT_TOKEN: Bot token for API authentication
- DISCORD_APPLICATION_ID: Application ID for interactions
- DISCORD_PUBLIC_KEY: Public key for webhook verification
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import httpx  # noqa: F401 - used for availability check and in download_file

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from nacl.signing import VerifyKey
    from nacl.exceptions import BadSignatureError

    NACL_AVAILABLE = True
except ImportError:
    NACL_AVAILABLE = False
    logger.warning("PyNaCl not available - Discord webhook verification disabled")

try:
    from aragora.observability.tracing import build_trace_headers
except ImportError:

    def build_trace_headers() -> dict[str, str]:
        return {}


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
    SendMessageResponse,
    UserInteraction,
    WebhookEvent,
)

# Environment configuration
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_APPLICATION_ID = os.environ.get("DISCORD_APPLICATION_ID", "")
DISCORD_PUBLIC_KEY = os.environ.get("DISCORD_PUBLIC_KEY", "")

# Discord API
DISCORD_API_BASE = "https://discord.com/api/v10"


class DiscordConnector(ChatPlatformConnector):
    """
    Discord connector using Discord API.

    Supports:
    - Sending messages with Embeds and Components
    - Slash commands and interactions
    - File uploads
    - Threaded messages (forum channels)
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        application_id: Optional[str] = None,
        public_key: Optional[str] = None,
        **config: Any,
    ):
        """
        Initialize Discord connector.

        Args:
            bot_token: Bot token (defaults to DISCORD_BOT_TOKEN)
            application_id: Application ID for interactions
            public_key: Public key for webhook verification
            **config: Additional configuration
        """
        super().__init__(
            bot_token=bot_token or DISCORD_BOT_TOKEN,
            signing_secret=public_key or DISCORD_PUBLIC_KEY,
            **config,
        )
        self.application_id = application_id or DISCORD_APPLICATION_ID
        self.public_key = public_key or DISCORD_PUBLIC_KEY

    @property
    def platform_name(self) -> str:
        return "discord"

    @property
    def platform_display_name(self) -> str:
        return "Discord"

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers with trace context for distributed tracing."""
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json",
        }
        # Add trace context headers for distributed tracing
        headers.update(build_trace_headers())
        return headers

    async def send_message(
        self,
        channel_id: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send message to Discord channel with circuit breaker protection."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(success=False, error=cb_error)

        try:
            # Build message payload
            payload: dict[str, Any] = {
                "content": text,
            }

            # Add embeds if blocks provided
            if blocks:
                payload["embeds"] = blocks

            # Add components (buttons) if provided
            components = kwargs.get("components")
            if components:
                payload["components"] = components

            # Handle thread
            target_channel = thread_id or channel_id

            # Use shared HTTP helper with retry and circuit breaker
            success, data, error = await self._http_request(
                method="POST",
                url=f"{DISCORD_API_BASE}/channels/{target_channel}/messages",
                headers=self._get_headers(),
                json=payload,
                operation="send_message",
            )

            if success and data:
                return SendMessageResponse(
                    success=True,
                    message_id=data.get("id"),
                    channel_id=data.get("channel_id"),
                    timestamp=data.get("timestamp"),
                )
            else:
                return SendMessageResponse(success=False, error=error or "Unknown error")

        except Exception as e:
            self._record_failure(e)
            logger.error(f"Discord send_message error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def update_message(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Update a Discord message with circuit breaker protection."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            return SendMessageResponse(success=False, error=cb_error)

        try:
            payload: dict[str, Any] = {
                "content": text,
            }

            if blocks:
                payload["embeds"] = blocks

            components = kwargs.get("components")
            if components:
                payload["components"] = components

            # Use shared HTTP helper with retry and circuit breaker
            success, _, error = await self._http_request(
                method="PATCH",
                url=f"{DISCORD_API_BASE}/channels/{channel_id}/messages/{message_id}",
                headers=self._get_headers(),
                json=payload,
                operation="update_message",
            )

            if success:
                return SendMessageResponse(
                    success=True,
                    message_id=message_id,
                    channel_id=channel_id,
                )
            else:
                return SendMessageResponse(success=False, error=error or "Unknown error")

        except Exception as e:
            self._record_failure(e)
            logger.error(f"Discord update_message error: {e}")
            return SendMessageResponse(success=False, error=str(e))

    async def delete_message(
        self,
        channel_id: str,
        message_id: str,
        **kwargs: Any,
    ) -> bool:
        """Delete a Discord message with retry, timeout, and circuit breaker protection."""
        if not HTTPX_AVAILABLE:
            return False

        try:
            # Use shared HTTP helper with retry and circuit breaker
            success, _, error = await self._http_request(
                method="DELETE",
                url=f"{DISCORD_API_BASE}/channels/{channel_id}/messages/{message_id}",
                headers=self._get_headers(),
                operation="delete_message",
            )
            result: bool = success
            return result

        except Exception as e:
            self._record_failure(e)
            logger.error(f"Discord delete_message error: {e}")
            return False

    async def send_typing_indicator(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> bool:
        """Send typing indicator to a Discord channel.

        Uses POST /channels/{channel_id}/typing endpoint.
        Typing indicators last for 10 seconds or until a message is sent.
        """
        if not HTTPX_AVAILABLE:
            return False

        try:
            success, _, _ = await self._http_request(
                method="POST",
                url=f"{DISCORD_API_BASE}/channels/{channel_id}/typing",
                headers=self._get_headers(),
                operation="send_typing",
            )
            result: bool = success
            return result

        except Exception as e:
            logger.debug(f"Discord typing indicator error: {e}")
            return False

    async def send_ephemeral(
        self,
        channel_id: str,
        user_id: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Send ephemeral message (only works in interaction responses)."""
        # Discord ephemeral messages only work in slash command/interaction responses
        # This is handled in respond_to_interaction
        logger.warning("Discord ephemeral messages require interaction context")
        return await self.send_message(channel_id, text, blocks, **kwargs)

    async def respond_to_command(
        self,
        command: BotCommand,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        ephemeral: bool = True,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Discord slash command."""
        if not command.metadata.get("interaction_token"):
            # No interaction token - send as regular message
            if command.channel:
                return await self.send_message(command.channel.id, text, blocks, **kwargs)
            return SendMessageResponse(success=False, error="No interaction token or channel")

        return await self._respond_to_interaction_token(
            command.metadata["interaction_id"],
            command.metadata["interaction_token"],
            text,
            blocks,
            ephemeral,
        )

    async def respond_to_interaction(
        self,
        interaction: UserInteraction,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        replace_original: bool = False,
        **kwargs: Any,
    ) -> SendMessageResponse:
        """Respond to a Discord interaction (button, select menu)."""
        interaction_id = interaction.metadata.get("interaction_id")
        interaction_token = interaction.metadata.get("interaction_token")

        if not interaction_id or not interaction_token:
            if interaction.channel:
                return await self.send_message(interaction.channel.id, text, blocks, **kwargs)
            return SendMessageResponse(success=False, error="No interaction context")

        # Use UPDATE_MESSAGE type if replacing original
        response_type = 7 if replace_original else 4

        return await self._respond_to_interaction_token(
            interaction_id,
            interaction_token,
            text,
            blocks,
            ephemeral=False,
            response_type=response_type,
        )

    async def _respond_to_interaction_token(
        self,
        interaction_id: str,
        interaction_token: str,
        text: str,
        blocks: Optional[list[dict[str, Any]]] = None,
        ephemeral: bool = False,
        response_type: int = 4,  # CHANNEL_MESSAGE_WITH_SOURCE
    ) -> SendMessageResponse:
        """Send response using interaction token with retry, timeout, and circuit breaker."""
        if not HTTPX_AVAILABLE:
            return SendMessageResponse(success=False, error="httpx not available")

        try:
            payload: dict[str, Any] = {
                "type": response_type,
                "data": {
                    "content": text,
                },
            }

            if blocks:
                payload["data"]["embeds"] = blocks

            if ephemeral:
                payload["data"]["flags"] = 64  # EPHEMERAL flag

            # Use shared HTTP helper with retry and circuit breaker
            # Note: Interaction callbacks have a 3-second window, so we use fewer retries
            success, _, error = await self._http_request(
                method="POST",
                url=f"{DISCORD_API_BASE}/interactions/{interaction_id}/{interaction_token}/callback",
                headers={"Content-Type": "application/json"},
                json=payload,
                max_retries=2,  # Reduced retries for time-sensitive interactions
                base_delay=0.5,
                operation="interaction_callback",
            )

            if success:
                return SendMessageResponse(success=True)
            else:
                return SendMessageResponse(success=False, error=error or "Unknown error")

        except Exception as e:
            self._record_failure(e)
            logger.error(f"Discord interaction response error: {e}")
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
        """Upload file to Discord channel with retry, timeout, and circuit breaker."""
        if not HTTPX_AVAILABLE:
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
            )

        try:
            target_channel = thread_id or channel_id

            # Discord uses multipart form for file uploads
            files = {"file": (filename, content, content_type)}
            data = {}
            if title:
                data["content"] = title

            # Use shared HTTP helper with retry and circuit breaker
            success, result, error = await self._http_request(
                method="POST",
                url=f"{DISCORD_API_BASE}/channels/{target_channel}/messages",
                headers={"Authorization": f"Bot {self.bot_token}"},
                files=files,
                data=data if data else None,
                operation="upload_file",
            )

            if success and result:
                attachments = result.get("attachments", [])
                if attachments:
                    att = attachments[0]
                    return FileAttachment(
                        id=att.get("id", ""),
                        filename=att.get("filename", filename),
                        content_type=att.get("content_type", content_type),
                        size=att.get("size", len(content)),
                        url=att.get("url"),
                    )

            logger.warning(f"Discord upload_file failed: {error}")
            return FileAttachment(
                id="",
                filename=filename,
                content_type=content_type,
                size=len(content),
            )

        except Exception as e:
            self._record_failure(e)
            logger.error(f"Discord upload_file error: {e}")
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
        """Download file from Discord (requires URL) with retry, timeout, and circuit breaker."""
        url = kwargs.get("url")
        if not url or not HTTPX_AVAILABLE:
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
            )

        try:
            # Use shared HTTP helper with retry and circuit breaker
            # Note: File downloads don't use auth headers
            success, result, error = await self._http_request(
                method="GET",
                url=url,
                operation="download_file",
            )

            if success and result:
                # The HTTP helper returns text in result["text"] for non-JSON responses
                # For binary content, we need to make a direct call
                import httpx

                can_proceed, cb_error = self._check_circuit_breaker()
                if not can_proceed:
                    return FileAttachment(
                        id=file_id,
                        filename="",
                        content_type="application/octet-stream",
                        size=0,
                    )

                async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                    self._record_success()
                    return FileAttachment(
                        id=file_id,
                        filename=kwargs.get("filename", "file"),
                        content_type=response.headers.get(
                            "content-type", "application/octet-stream"
                        ),
                        size=len(response.content),
                        url=url,
                        content=response.content,
                    )

            logger.warning(f"Discord download_file failed: {error}")
            return FileAttachment(
                id=file_id,
                filename="",
                content_type="application/octet-stream",
                size=0,
            )

        except Exception as e:
            self._record_failure(e)
            logger.error(f"Discord download_file error: {e}")
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
    ) -> list[dict[str, Any]]:
        """Format content as Discord Embed."""
        embed: dict[str, Any] = {
            "type": "rich",
        }

        if title:
            embed["title"] = title

        if body:
            embed["description"] = body

        if fields:
            embed["fields"] = [
                {"name": label, "value": value, "inline": True} for label, value in fields
            ]

        color = kwargs.get("color", 0x00FF00)  # Default: green
        embed["color"] = color

        return [embed]

    def format_button(
        self,
        text: str,
        action_id: str,
        value: Optional[str] = None,
        style: str = "default",
        url: Optional[str] = None,
    ) -> dict[str, Any]:
        """Format a Discord button component."""
        if url:
            return {
                "type": 2,  # BUTTON
                "style": 5,  # LINK
                "label": text,
                "url": url,
            }

        style_map = {
            "default": 2,  # SECONDARY
            "primary": 1,  # PRIMARY
            "danger": 4,  # DANGER
        }

        return {
            "type": 2,  # BUTTON
            "style": style_map.get(style, 2),
            "label": text,
            "custom_id": f"{action_id}:{value or ''}",
        }

    def verify_webhook(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> bool:
        """Verify Discord interaction webhook signature.

        SECURITY: Fails closed in production if PyNaCl is unavailable or public_key not configured.
        Uses centralized webhook_security module for production-safe bypass handling.
        """
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        if not NACL_AVAILABLE:
            # SECURITY: Use centralized bypass check (ignores flag in production)
            if should_allow_unverified("discord"):
                logger.warning(
                    "Discord webhook verification skipped - PyNaCl not available (dev mode)"
                )
                return True
            logger.error("Discord webhook rejected - PyNaCl not available")
            return False

        if not self.public_key:
            if should_allow_unverified("discord"):
                logger.warning(
                    "Discord webhook verification skipped - public_key not configured (dev mode)"
                )
                return True
            logger.error("Discord webhook rejected - public_key not configured")
            return False

        signature = headers.get("X-Signature-Ed25519", "")
        timestamp = headers.get("X-Signature-Timestamp", "")

        if not signature or not timestamp:
            return False

        try:
            verify_key = VerifyKey(bytes.fromhex(self.public_key))
            verify_key.verify(f"{timestamp}{body.decode()}".encode(), bytes.fromhex(signature))
            return True
        except BadSignatureError:
            return False
        except Exception as e:
            logger.error(f"Discord signature verification error: {e}")
            return False

    def parse_webhook_event(
        self,
        headers: dict[str, str],
        body: bytes,
    ) -> WebhookEvent:
        """Parse Discord interaction into WebhookEvent."""
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return WebhookEvent(
                platform=self.platform_name,
                event_type="error",
                raw_payload={},
            )

        interaction_type = payload.get("type", 0)

        # Type 1: PING (URL verification)
        if interaction_type == 1:
            return WebhookEvent(
                platform=self.platform_name,
                event_type="ping",
                raw_payload=payload,
                challenge="PONG",
            )

        # Parse user
        user_data = payload.get("member", {}).get("user", {}) or payload.get("user", {})
        user = ChatUser(
            id=user_data.get("id", ""),
            platform=self.platform_name,
            username=user_data.get("username"),
            display_name=user_data.get("global_name") or user_data.get("username"),
            avatar_url=(
                f"https://cdn.discordapp.com/avatars/{user_data.get('id')}/{user_data.get('avatar')}.png"
                if user_data.get("avatar")
                else None
            ),
            is_bot=user_data.get("bot", False),
        )

        # Parse channel
        channel = ChatChannel(
            id=payload.get("channel_id", ""),
            platform=self.platform_name,
            team_id=payload.get("guild_id"),
        )

        event = WebhookEvent(
            platform=self.platform_name,
            event_type=f"interaction_{interaction_type}",
            raw_payload=payload,
            metadata={
                "interaction_id": payload.get("id"),
                "interaction_token": payload.get("token"),
            },
        )

        # Type 2: APPLICATION_COMMAND (slash command)
        if interaction_type == 2:
            cmd_data = payload.get("data", {})
            options = cmd_data.get("options", [])

            event.command = BotCommand(
                name=cmd_data.get("name", ""),
                text=f"/{cmd_data.get('name', '')}",
                args=[opt.get("value") for opt in options if opt.get("value")],
                options={opt.get("name"): opt.get("value") for opt in options},
                user=user,
                channel=channel,
                platform=self.platform_name,
                metadata={
                    "interaction_id": payload.get("id"),
                    "interaction_token": payload.get("token"),
                },
            )

        # Type 3: MESSAGE_COMPONENT (button click, select menu)
        elif interaction_type == 3:
            comp_data = payload.get("data", {})
            custom_id = comp_data.get("custom_id", "")

            # Parse action_id:value format
            parts = custom_id.split(":", 1)
            action_id = parts[0]
            value = parts[1] if len(parts) > 1 else None

            comp_type = comp_data.get("component_type", 2)
            int_type = (
                InteractionType.BUTTON_CLICK if comp_type == 2 else InteractionType.SELECT_MENU
            )

            event.interaction = UserInteraction(
                id=payload.get("id", ""),
                interaction_type=int_type,
                action_id=action_id,
                value=value,
                values=comp_data.get("values", []),
                user=user,
                channel=channel,
                message_id=payload.get("message", {}).get("id"),
                platform=self.platform_name,
                metadata={
                    "interaction_id": payload.get("id"),
                    "interaction_token": payload.get("token"),
                },
            )

        # Type 5: MODAL_SUBMIT
        elif interaction_type == 5:
            comp_data = payload.get("data", {})

            event.interaction = UserInteraction(
                id=payload.get("id", ""),
                interaction_type=InteractionType.MODAL_SUBMIT,
                action_id=comp_data.get("custom_id", ""),
                user=user,
                channel=channel,
                platform=self.platform_name,
                metadata={
                    "interaction_id": payload.get("id"),
                    "interaction_token": payload.get("token"),
                    "components": comp_data.get("components", []),
                },
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
        Get message history from a Discord channel with retry, timeout, and circuit breaker.

        Uses Discord's GET /channels/{channel.id}/messages API.

        Args:
            channel_id: Discord channel ID
            limit: Maximum number of messages (max 100 per request)
            oldest: Get messages after this message ID
            latest: Get messages before this message ID
            **kwargs: Additional options

        Returns:
            List of ChatMessage objects
        """
        if not HTTPX_AVAILABLE:
            logger.error("httpx not available for Discord API")
            return []

        try:
            # Build query string for params
            params_str = f"?limit={min(limit, 100)}"
            if oldest:
                params_str += f"&after={oldest}"
            if latest:
                params_str += f"&before={latest}"

            # Use shared HTTP helper with retry and circuit breaker
            success, data, error = await self._http_request(
                method="GET",
                url=f"{DISCORD_API_BASE}/channels/{channel_id}/messages{params_str}",
                headers=self._get_headers(),
                operation="get_channel_history",
            )

            if not success or not data:
                logger.error(f"Discord API error: {error}")
                return []

            # Handle case where response is a list (normal) vs dict (error)
            if isinstance(data, dict) and "text" in data:
                # Try to parse the text as JSON list
                try:
                    data = json.loads(data["text"])
                except (json.JSONDecodeError, KeyError):
                    logger.error(f"Could not parse Discord response: {data}")
                    return []
            elif not isinstance(data, list):
                logger.error(f"Unexpected Discord response type: {type(data)}")
                return []

            messages: list[ChatMessage] = []

            channel = ChatChannel(
                id=channel_id,
                platform=self.platform_name,
            )

            for msg in data:
                # Skip bot messages if configured
                if kwargs.get("skip_bots", True) and msg.get("author", {}).get("bot"):
                    continue

                author_data = msg.get("author", {})
                user = ChatUser(
                    id=author_data.get("id", ""),
                    platform=self.platform_name,
                    username=author_data.get("username"),
                    display_name=author_data.get("global_name"),
                    avatar_url=(
                        f"https://cdn.discordapp.com/avatars/{author_data.get('id')}/{author_data.get('avatar')}.png"
                        if author_data.get("avatar")
                        else None
                    ),
                    is_bot=author_data.get("bot", False),
                )

                # Parse timestamp
                timestamp_str = msg.get("timestamp", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    timestamp = datetime.now(timezone.utc)

                chat_msg = ChatMessage(
                    id=msg.get("id", ""),
                    platform=self.platform_name,
                    channel=channel,
                    author=user,
                    content=msg.get("content", ""),
                    thread_id=msg.get("message_reference", {}).get("message_id"),
                    timestamp=timestamp,
                    metadata={
                        "reactions": msg.get("reactions", []),
                        "attachments": msg.get("attachments", []),
                        "embeds": msg.get("embeds", []),
                    },
                )
                messages.append(chat_msg)

            return messages

        except Exception as e:
            self._record_failure(e)
            logger.error(f"Error getting Discord channel history: {e}")
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

        Retrieves messages from a Discord channel, filters by relevance,
        and converts to ChatEvidence format.

        Args:
            channel_id: Discord channel ID
            query: Optional search query to filter messages
            limit: Maximum number of messages
            include_threads: Whether to include thread messages
            min_relevance: Minimum relevance score for inclusion
            **kwargs: Additional options

        Returns:
            List of ChatEvidence objects
        """
        messages = await self.get_channel_history(
            channel_id=channel_id,
            limit=limit,
            **kwargs,
        )

        if not messages:
            return []

        evidence_list: list[ChatEvidence] = []

        for msg in messages:
            relevance = self._compute_message_relevance(msg, query)

            if relevance < min_relevance:
                continue

            evidence = ChatEvidence.from_message(
                message=msg,
                query=query,
                relevance_score=relevance,
            )

            evidence_list.append(evidence)

        # Sort by relevance
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)

        logger.info(
            f"Collected {len(evidence_list)} evidence items from Discord channel {channel_id}"
        )

        return evidence_list

    async def get_channel_info(
        self,
        channel_id: str,
        **kwargs: Any,
    ) -> Optional[ChatChannel]:
        """
        Get information about a Discord channel.

        Uses Discord's GET /channels/{channel.id} API.

        Args:
            channel_id: Discord channel ID
            **kwargs: Additional options

        Returns:
            ChatChannel info or None
        """
        if not HTTPX_AVAILABLE:
            logger.debug("httpx not available for Discord API")
            return None

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            logger.debug(f"Circuit breaker open: {cb_error}")
            return None

        try:
            success, data, error = await self._http_request(
                method="GET",
                url=f"{DISCORD_API_BASE}/channels/{channel_id}",
                headers=self._get_headers(),
                operation="get_channel_info",
            )

            if not success or not data:
                logger.debug(f"Failed to get channel info: {error}")
                return None

            # Channel type mapping
            # 0: GUILD_TEXT, 1: DM, 2: GUILD_VOICE, 3: GROUP_DM
            # 4: GUILD_CATEGORY, 5: GUILD_ANNOUNCEMENT, 10: ANNOUNCEMENT_THREAD
            # 11: PUBLIC_THREAD, 12: PRIVATE_THREAD, 13: GUILD_STAGE_VOICE
            # 14: GUILD_DIRECTORY, 15: GUILD_FORUM
            channel_type = data.get("type", 0)
            is_dm = channel_type in (1, 3)  # DM or Group DM
            is_private = channel_type == 12  # Private thread

            return ChatChannel(
                id=channel_id,
                platform=self.platform_name,
                name=data.get("name"),
                is_private=is_private,
                is_dm=is_dm,
                team_id=data.get("guild_id"),
                metadata={
                    "type": channel_type,
                    "topic": data.get("topic"),
                    "nsfw": data.get("nsfw", False),
                    "position": data.get("position"),
                    "parent_id": data.get("parent_id"),
                    "rate_limit_per_user": data.get("rate_limit_per_user"),
                },
            )

        except Exception as e:
            logger.debug(f"Discord get_channel_info error: {e}")
            return None

    async def get_user_info(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> Optional[ChatUser]:
        """
        Get information about a Discord user.

        Uses Discord's GET /users/{user.id} API.

        Args:
            user_id: Discord user ID
            **kwargs: Additional options

        Returns:
            ChatUser info or None
        """
        if not HTTPX_AVAILABLE:
            logger.debug("httpx not available for Discord API")
            return None

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            logger.debug(f"Circuit breaker open: {cb_error}")
            return None

        try:
            success, data, error = await self._http_request(
                method="GET",
                url=f"{DISCORD_API_BASE}/users/{user_id}",
                headers=self._get_headers(),
                operation="get_user_info",
            )

            if not success or not data:
                logger.debug(f"Failed to get user info: {error}")
                return None

            # Build avatar URL
            avatar_hash = data.get("avatar")
            avatar_url = None
            if avatar_hash:
                # Use .gif for animated avatars (start with a_)
                extension = "gif" if avatar_hash.startswith("a_") else "png"
                avatar_url = (
                    f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.{extension}"
                )

            return ChatUser(
                id=user_id,
                platform=self.platform_name,
                username=data.get("username"),
                display_name=data.get("global_name") or data.get("username"),
                avatar_url=avatar_url,
                is_bot=data.get("bot", False),
                metadata={
                    "discriminator": data.get("discriminator"),
                    "accent_color": data.get("accent_color"),
                    "banner": data.get("banner"),
                    "public_flags": data.get("public_flags"),
                },
            )

        except Exception as e:
            logger.debug(f"Discord get_user_info error: {e}")
            return None
