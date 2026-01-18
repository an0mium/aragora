"""
Slack Enterprise Connector.

Provides full integration with Slack workspaces:
- Channel message indexing
- Thread extraction
- File attachment handling
- User mention resolution
- Incremental sync via timestamps
- Webhook/Events API support

Requires Slack Bot Token with appropriate scopes.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


@dataclass
class SlackChannel:
    """A Slack channel."""

    id: str
    name: str
    is_private: bool = False
    is_archived: bool = False
    topic: str = ""
    purpose: str = ""
    member_count: int = 0
    created: Optional[datetime] = None


@dataclass
class SlackMessage:
    """A Slack message."""

    ts: str  # Timestamp (unique ID)
    channel_id: str
    text: str
    user_id: str = ""
    user_name: str = ""
    thread_ts: Optional[str] = None
    reply_count: int = 0
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None


@dataclass
class SlackUser:
    """A Slack user."""

    id: str
    name: str
    real_name: str = ""
    display_name: str = ""
    email: str = ""
    is_bot: bool = False


class SlackConnector(EnterpriseConnector):
    """
    Enterprise connector for Slack workspaces.

    Features:
    - Public and private channel indexing
    - Message and thread extraction
    - File metadata indexing
    - User mention resolution
    - Reaction tracking
    - Incremental sync via message timestamps
    - Real-time updates via Events API

    Authentication:
    - Bot Token with scopes:
      - channels:history, channels:read
      - groups:history, groups:read (for private channels)
      - users:read
      - files:read

    Usage:
        connector = SlackConnector(
            workspace_name="MyCompany",
            channels=["engineering", "general"],  # Optional: specific channels
        )
        result = await connector.sync()
    """

    def __init__(
        self,
        workspace_name: str = "default",
        channels: Optional[List[str]] = None,
        include_private: bool = False,
        include_archived: bool = False,
        include_threads: bool = True,
        include_files: bool = True,
        exclude_bots: bool = True,
        max_messages_per_channel: int = 1000,
        **kwargs,
    ):
        """
        Initialize Slack connector.

        Args:
            workspace_name: Name for identification
            channels: Specific channel names to sync (None = all accessible)
            include_private: Whether to include private channels
            include_archived: Whether to include archived channels
            include_threads: Whether to fetch thread replies
            include_files: Whether to index file metadata
            exclude_bots: Whether to exclude bot messages
            max_messages_per_channel: Maximum messages to fetch per channel
        """
        connector_id = f"slack_{workspace_name.lower().replace(' ', '_')}"
        super().__init__(connector_id=connector_id, **kwargs)

        self.workspace_name = workspace_name
        self.channels = set(channels) if channels else None
        self.include_private = include_private
        self.include_archived = include_archived
        self.include_threads = include_threads
        self.include_files = include_files
        self.exclude_bots = exclude_bots
        self.max_messages_per_channel = max_messages_per_channel

        # Cache
        self._users_cache: Dict[str, SlackUser] = {}
        self._channels_cache: Dict[str, SlackChannel] = {}

    @property
    def source_type(self) -> SourceType:
        return SourceType.SYNTHESIS  # Conversations combine multiple contributors

    @property
    def name(self) -> str:
        return f"Slack ({self.workspace_name})"

    async def _get_auth_header(self) -> Dict[str, str]:
        """Get authentication header."""
        token = await self.credentials.get_credential("SLACK_BOT_TOKEN")

        if not token:
            raise ValueError(
                "Slack credentials not configured. Set SLACK_BOT_TOKEN"
            )

        return {"Authorization": f"Bearer {token}"}

    async def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a request to Slack Web API."""
        import httpx

        headers = await self._get_auth_header()
        headers["Content-Type"] = "application/json; charset=utf-8"

        url = f"https://slack.com/api/{endpoint}"

        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params, timeout=60)
            else:
                response = await client.post(url, headers=headers, json=json_data or params, timeout=60)

            response.raise_for_status()
            data = response.json()

            if not data.get("ok"):
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Slack API error: {error}")

            return data

    async def _get_channels(self) -> List[SlackChannel]:
        """Get all accessible channels."""
        channels = []
        cursor = None

        while True:
            params: Dict[str, Any] = {
                "limit": 200,
                "exclude_archived": not self.include_archived,
            }

            if cursor:
                params["cursor"] = cursor

            # Get public channels
            data = await self._api_request("conversations.list", params=params)

            for item in data.get("channels", []):
                channel_name = item.get("name", "")

                # Filter to specific channels if configured
                if self.channels and channel_name not in self.channels:
                    continue

                # Skip private channels if not included
                if item.get("is_private") and not self.include_private:
                    continue

                channel = SlackChannel(
                    id=item.get("id", ""),
                    name=channel_name,
                    is_private=item.get("is_private", False),
                    is_archived=item.get("is_archived", False),
                    topic=item.get("topic", {}).get("value", ""),
                    purpose=item.get("purpose", {}).get("value", ""),
                    member_count=item.get("num_members", 0),
                    created=datetime.fromtimestamp(
                        item.get("created", 0), tz=timezone.utc
                    )
                    if item.get("created")
                    else None,
                )
                channels.append(channel)
                self._channels_cache[channel.id] = channel

            # Check pagination
            cursor = data.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        return channels

    async def _get_user(self, user_id: str) -> Optional[SlackUser]:
        """Get user info by ID."""
        if user_id in self._users_cache:
            return self._users_cache[user_id]

        try:
            data = await self._api_request("users.info", params={"user": user_id})
            user_data = data.get("user", {})

            user = SlackUser(
                id=user_data.get("id", ""),
                name=user_data.get("name", ""),
                real_name=user_data.get("real_name", ""),
                display_name=user_data.get("profile", {}).get("display_name", ""),
                email=user_data.get("profile", {}).get("email", ""),
                is_bot=user_data.get("is_bot", False),
            )
            self._users_cache[user_id] = user
            return user

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get user {user_id}: {e}")
            return None

    async def _get_messages(
        self,
        channel_id: str,
        oldest: Optional[str] = None,
        limit: int = 100,
    ) -> tuple[List[SlackMessage], bool]:
        """Get messages from a channel."""
        params: Dict[str, Any] = {
            "channel": channel_id,
            "limit": min(limit, 200),
        }

        if oldest:
            params["oldest"] = oldest

        data = await self._api_request("conversations.history", params=params)

        messages = []
        for item in data.get("messages", []):
            # Skip bot messages if configured
            if self.exclude_bots and (item.get("bot_id") or item.get("subtype") == "bot_message"):
                continue

            # Parse timestamp to datetime
            ts = item.get("ts", "")
            created_at = None
            if ts:
                try:
                    created_at = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Invalid Slack timestamp format: {e}")

            message = SlackMessage(
                ts=ts,
                channel_id=channel_id,
                text=item.get("text", ""),
                user_id=item.get("user", ""),
                thread_ts=item.get("thread_ts") if item.get("reply_count") else None,
                reply_count=item.get("reply_count", 0),
                reactions=item.get("reactions", []),
                files=item.get("files", []),
                created_at=created_at,
            )
            messages.append(message)

        has_more = data.get("has_more", False)
        return messages, has_more

    async def _get_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
    ) -> List[SlackMessage]:
        """Get replies in a thread."""
        if not self.include_threads:
            return []

        try:
            params = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": 100,
            }

            data = await self._api_request("conversations.replies", params=params)

            replies = []
            for item in data.get("messages", [])[1:]:  # Skip first (parent)
                if self.exclude_bots and (item.get("bot_id") or item.get("subtype") == "bot_message"):
                    continue

                ts = item.get("ts", "")
                created_at = None
                if ts:
                    try:
                        created_at = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Invalid Slack thread timestamp: {e}")

                replies.append(
                    SlackMessage(
                        ts=ts,
                        channel_id=channel_id,
                        text=item.get("text", ""),
                        user_id=item.get("user", ""),
                        thread_ts=thread_ts,
                        created_at=created_at,
                    )
                )

            return replies

        except Exception as e:
            logger.warning(f"[{self.name}] Failed to get thread replies: {e}")
            return []

    async def _resolve_mentions(self, text: str) -> str:
        """Resolve user mentions in text."""
        # Find all user mentions
        mentions = re.findall(r"<@([A-Z0-9]+)>", text)

        for user_id in set(mentions):
            user = await self._get_user(user_id)
            if user:
                display_name = user.display_name or user.real_name or user.name
                text = text.replace(f"<@{user_id}>", f"@{display_name}")

        # Clean up channel mentions
        text = re.sub(r"<#([A-Z0-9]+)\|([^>]+)>", r"#\2", text)

        # Clean up URLs
        text = re.sub(r"<(https?://[^|>]+)\|([^>]+)>", r"\2 (\1)", text)
        text = re.sub(r"<(https?://[^>]+)>", r"\1", text)

        return text

    def _format_message_content(
        self,
        message: SlackMessage,
        channel: SlackChannel,
        user: Optional[SlackUser],
        replies: List[SlackMessage] = None,
    ) -> str:
        """Format a message with context."""
        parts = []

        # Header
        user_name = ""
        if user:
            user_name = user.display_name or user.real_name or user.name
        timestamp = message.created_at.strftime("%Y-%m-%d %H:%M") if message.created_at else ""

        parts.append(f"#{channel.name} | {user_name} | {timestamp}")
        parts.append("")
        parts.append(message.text)

        # Reactions
        if message.reactions:
            reaction_strs = []
            for reaction in message.reactions:
                emoji = reaction.get("name", "")
                count = reaction.get("count", 1)
                reaction_strs.append(f":{emoji}: ({count})")
            parts.append(f"\nReactions: {' '.join(reaction_strs)}")

        # Files
        if message.files and self.include_files:
            parts.append("\nAttachments:")
            for f in message.files:
                name = f.get("name", "file")
                filetype = f.get("filetype", "")
                parts.append(f"  - {name} ({filetype})")

        # Thread replies
        if replies:
            parts.append(f"\nThread ({len(replies)} replies):")
            for reply in replies[:10]:  # Limit to 10 replies
                reply_user = self._users_cache.get(reply.user_id)
                reply_name = ""
                if reply_user:
                    reply_name = reply_user.display_name or reply_user.real_name or reply_user.name
                parts.append(f"  {reply_name}: {reply.text[:200]}")

        return "\n".join(parts)

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield Slack messages for syncing.
        """
        # Parse last sync timestamp from cursor
        oldest_ts = state.cursor

        # Get all channels
        channels = await self._get_channels()
        state.items_total = len(channels)

        items_yielded = 0

        for channel in channels:
            logger.info(f"[{self.name}] Syncing channel: #{channel.name}")

            messages_fetched = 0
            current_oldest = oldest_ts

            while messages_fetched < self.max_messages_per_channel:
                messages, has_more = await self._get_messages(
                    channel.id,
                    oldest=current_oldest,
                    limit=min(100, self.max_messages_per_channel - messages_fetched),
                )

                if not messages:
                    break

                for message in messages:
                    # Get user info
                    user = None
                    if message.user_id:
                        user = await self._get_user(message.user_id)
                        message.user_name = user.display_name if user else ""

                    # Resolve mentions in text
                    resolved_text = await self._resolve_mentions(message.text)
                    message.text = resolved_text

                    # Get thread replies if this is a parent message
                    replies = []
                    if message.thread_ts == message.ts and message.reply_count > 0:
                        replies = await self._get_thread_replies(channel.id, message.ts)

                    # Format content
                    content = self._format_message_content(message, channel, user, replies)

                    yield SyncItem(
                        id=f"slack-{channel.id}-{message.ts}",
                        content=content[:50000],
                        source_type="discussion",
                        source_id=f"slack/{self.workspace_name}/{channel.name}/{message.ts}",
                        title=f"#{channel.name} - {message.user_name}",
                        url=f"https://slack.com/archives/{channel.id}/p{message.ts.replace('.', '')}",
                        author=message.user_name,
                        created_at=message.created_at,
                        domain="enterprise/slack",
                        confidence=0.75,
                        metadata={
                            "channel_id": channel.id,
                            "channel_name": channel.name,
                            "user_id": message.user_id,
                            "ts": message.ts,
                            "thread_ts": message.thread_ts,
                            "reply_count": message.reply_count,
                            "reaction_count": sum(r.get("count", 0) for r in message.reactions),
                            "file_count": len(message.files),
                        },
                    )

                    items_yielded += 1

                    # Update cursor to latest timestamp
                    if not state.cursor or message.ts > state.cursor:
                        state.cursor = message.ts

                messages_fetched += len(messages)

                if not has_more:
                    break

                # Update oldest for next batch
                if messages:
                    current_oldest = messages[-1].ts

                await asyncio.sleep(0.5)  # Rate limiting

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list:
        """Search Slack messages."""
        from aragora.connectors.base import Evidence

        try:
            data = await self._api_request(
                "search.messages",
                params={
                    "query": query,
                    "count": limit,
                    "sort": "score",
                },
            )

            results = []
            for match in data.get("messages", {}).get("matches", []):
                channel_id = match.get("channel", {}).get("id", "")
                channel_name = match.get("channel", {}).get("name", "")
                ts = match.get("ts", "")
                user_name = match.get("username", "")

                results.append(
                    Evidence(
                        id=f"slack-{channel_id}-{ts}",
                        source_type=self.source_type,
                        source_id=f"slack/{channel_name}/{ts}",
                        content=match.get("text", "")[:2000],
                        title=f"#{channel_name} - {user_name}",
                        url=match.get("permalink", ""),
                        author=user_name,
                        confidence=0.75,
                        metadata={
                            "channel_id": channel_id,
                            "channel_name": channel_name,
                            "score": match.get("score", 0),
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"[{self.name}] Search failed: {e}")
            return []

    async def fetch(self, evidence_id: str) -> Optional[Any]:
        """Fetch a specific Slack message."""
        from aragora.connectors.base import Evidence

        # Parse evidence_id: slack-{channel_id}-{ts}
        parts = evidence_id.split("-")
        if len(parts) < 3:
            return None

        channel_id = parts[1]
        ts = "-".join(parts[2:])  # Timestamp may contain dashes

        try:
            data = await self._api_request(
                "conversations.history",
                params={
                    "channel": channel_id,
                    "latest": ts,
                    "inclusive": True,
                    "limit": 1,
                },
            )

            messages = data.get("messages", [])
            if not messages:
                return None

            message = messages[0]
            text = await self._resolve_mentions(message.get("text", ""))

            channel = self._channels_cache.get(channel_id)
            channel_name = channel.name if channel else channel_id

            return Evidence(
                id=evidence_id,
                source_type=self.source_type,
                source_id=f"slack/{channel_name}/{ts}",
                content=text,
                title=f"#{channel_name}",
                url=f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}",
                author=message.get("user", ""),
                confidence=0.75,
            )

        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return None

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle Slack Events API webhook."""
        # URL verification challenge
        if payload.get("type") == "url_verification":
            return True

        # Event callback
        if payload.get("type") == "event_callback":
            event = payload.get("event", {})
            event_type = event.get("type", "")

            logger.info(f"[{self.name}] Webhook event: {event_type}")

            if event_type in ["message", "message.channels", "message.groups"]:
                # New message - trigger incremental sync
                asyncio.create_task(self.sync(max_items=10))
                return True

        return False


__all__ = ["SlackConnector", "SlackChannel", "SlackMessage", "SlackUser"]
