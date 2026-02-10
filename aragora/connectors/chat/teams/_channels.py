"""
Microsoft Teams channel operations mixin.

Provides channel history retrieval, evidence collection, channel/user
info lookups for the TeamsConnector.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Protocol

from aragora.connectors.chat.models import (
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
)

import aragora.connectors.chat.teams._constants as _tc

try:
    import httpx
except ImportError:
    pass

logger = logging.getLogger(__name__)

_MAX_PAGES = 1000  # Safety cap for pagination loops


class _TeamsConnectorProtocol(Protocol):
    """Protocol for methods expected by TeamsChannelsMixin from the main connector."""

    @property
    def platform_name(self) -> str: ...

    def _check_circuit_breaker(self) -> tuple[bool, str | None]: ...

    async def _graph_api_request(
        self,
        endpoint: str,
        method: str = ...,
        operation: str = ...,
    ) -> tuple[bool, dict[str, Any] | None, str | None]: ...

    def _record_failure(self, error: Exception | None = ...) -> None: ...

    def _compute_message_relevance(self, msg: ChatMessage, query: str | None) -> float: ...

    async def get_channel_history(
        self,
        channel_id: str,
        limit: int = ...,
        team_id: str | None = ...,
        **kwargs: Any,
    ) -> list[ChatMessage]: ...


class TeamsChannelsMixin:
    """Mixin providing channel operations for TeamsConnector."""

    async def get_channel_history(
        self: _TeamsConnectorProtocol,
        channel_id: str,
        limit: int = 100,
        oldest: str | None = None,
        latest: str | None = None,
        team_id: str | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """
        Get message history from a Teams channel via Microsoft Graph API.

        Uses the channelMessages API to retrieve messages.
        Requires ChannelMessage.Read.All permission.

        Args:
            channel_id: Teams channel ID
            limit: Maximum number of messages (max 50 per request)
            oldest: ISO timestamp - messages after this time
            latest: ISO timestamp - messages before this time
            team_id: Team ID (required for Graph API)
            **kwargs: Additional options

        Returns:
            List of ChatMessage objects
        """
        if not _tc.HTTPX_AVAILABLE:
            logger.error("httpx not available for Graph API")
            return []

        actual_team_id = team_id or kwargs.get("team_id")
        if not actual_team_id:
            logger.error("Team ID required for get_channel_history")
            return []

        # Check circuit breaker
        can_proceed, cb_error = self._check_circuit_breaker()
        if not can_proceed:
            logger.warning(f"Circuit breaker open: {cb_error}")
            return []

        try:
            messages: list[ChatMessage] = []
            next_link: str | None = None

            # Build initial endpoint with filters
            endpoint = f"/teams/{actual_team_id}/channels/{channel_id}/messages"
            params = [f"$top={min(limit, 50)}"]  # Graph API max is 50 per page

            if oldest:
                params.append(f"$filter=createdDateTime gt {oldest}")

            if params:
                endpoint = f"{endpoint}?{'&'.join(params)}"

            for _page in range(_MAX_PAGES):
                if next_link:
                    # Use the full nextLink URL directly
                    success, data, error = await self._graph_api_request(
                        endpoint=next_link.replace(_tc.GRAPH_API_BASE, ""),
                        method="GET",
                        operation="get_channel_messages",
                    )
                else:
                    success, data, error = await self._graph_api_request(
                        endpoint=endpoint,
                        method="GET",
                        operation="get_channel_messages",
                    )

                if not success or not data:
                    logger.error(f"Failed to get channel messages: {error}")
                    break

                # Parse messages from response
                for msg_data in data.get("value", []):
                    msg_id = msg_data.get("id", "")
                    created_at_str = msg_data.get("createdDateTime", "")

                    # Parse timestamp
                    try:
                        timestamp = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        timestamp = datetime.utcnow()

                    # Filter by latest timestamp if provided
                    if latest:
                        try:
                            latest_dt = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                            if timestamp > latest_dt:
                                continue
                        except (ValueError, AttributeError):
                            pass

                    # Parse author
                    from_data = msg_data.get("from", {}) or {}
                    user_data = from_data.get("user", {}) or {}
                    author = ChatUser(
                        id=user_data.get("id", ""),
                        platform=self.platform_name,
                        display_name=user_data.get("displayName"),
                        metadata={"aadObjectId": user_data.get("aadObjectId")},
                    )

                    # Parse channel info
                    channel = ChatChannel(
                        id=channel_id,
                        platform=self.platform_name,
                        team_id=actual_team_id,
                    )

                    # Extract message content
                    body = msg_data.get("body", {}) or {}
                    content = body.get("content", "")

                    # Strip HTML if content type is html
                    if body.get("contentType") == "html":
                        import re

                        content = re.sub(r"<[^>]+>", "", content)

                    messages.append(
                        ChatMessage(
                            id=msg_id,
                            platform=self.platform_name,
                            channel=channel,
                            author=author,
                            content=content,
                            timestamp=timestamp,
                            thread_id=msg_data.get("replyToId"),
                            metadata={
                                "importance": msg_data.get("importance"),
                                "web_url": msg_data.get("webUrl"),
                            },
                        )
                    )

                    if len(messages) >= limit:
                        break

                # Check for more pages
                next_link = data.get("@odata.nextLink")
                if not next_link or len(messages) >= limit:
                    break
            else:
                logger.warning("Pagination safety cap reached for Teams channel messages")

            logger.debug(f"Retrieved {len(messages)} messages from Teams channel {channel_id}")
            return messages[:limit]

        except httpx.TimeoutException as e:
            classified = _tc._classify_teams_error(f"Timeout: {e}")
            logger.error(f"Teams get_channel_history timeout: {e}")
            self._record_failure(classified)
            return []
        except httpx.ConnectError as e:
            classified = _tc._classify_teams_error(f"Connection error: {e}")
            logger.error(f"Teams get_channel_history connection error: {e}")
            self._record_failure(classified)
            return []
        except (
            httpx.HTTPError,
            RuntimeError,
            KeyError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ) as e:
            classified = _tc._classify_teams_error(str(e))
            logger.error(f"Teams get_channel_history error: {e}")
            self._record_failure(classified)
            return []

    async def collect_evidence(
        self: _TeamsConnectorProtocol,
        channel_id: str,
        query: str | None = None,
        limit: int = 100,
        include_threads: bool = True,
        min_relevance: float = 0.0,
        team_id: str | None = None,
        **kwargs: Any,
    ) -> list[ChatEvidence]:
        """
        Collect chat messages as evidence for debates.

        Retrieves messages from a Teams channel, filters by relevance,
        and converts to ChatEvidence format with provenance tracking.

        Args:
            channel_id: Teams channel ID
            query: Optional search query to filter messages
            limit: Maximum number of messages to retrieve
            include_threads: Whether to include reply messages
            min_relevance: Minimum relevance score for inclusion (0-1)
            team_id: Team ID (required for Graph API)
            **kwargs: Additional options

        Returns:
            List of ChatEvidence objects with relevance scoring
        """
        # Get channel history
        messages = await self.get_channel_history(
            channel_id=channel_id,
            limit=limit,
            team_id=team_id or kwargs.get("team_id"),
            **kwargs,
        )

        if not messages:
            return []

        # Convert to evidence with relevance scoring
        evidence_list: list[ChatEvidence] = []

        for msg in messages:
            # Skip replies if not including threads
            if not include_threads and msg.thread_id:
                continue

            # Calculate relevance using base class helper
            relevance = self._compute_message_relevance(msg, query)

            # Apply minimum relevance filter
            if relevance < min_relevance:
                continue

            # Convert to ChatEvidence
            evidence = ChatEvidence.from_message(
                message=msg,
                query=query or "",
                relevance_score=relevance,
            )

            evidence_list.append(evidence)

        # Sort by relevance score (highest first)
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)

        logger.debug(
            f"Collected {len(evidence_list)} evidence items from Teams channel {channel_id}"
        )
        return evidence_list

    async def get_channel_info(
        self: _TeamsConnectorProtocol,
        channel_id: str,
        team_id: str | None = None,
        **kwargs: Any,
    ) -> ChatChannel | None:
        """
        Get information about a Teams channel via Microsoft Graph API.

        Args:
            channel_id: Channel ID
            team_id: Team ID (required for Graph API)
            **kwargs: Additional options

        Returns:
            ChatChannel info or None
        """
        actual_team_id = team_id or kwargs.get("team_id")
        if not actual_team_id:
            logger.debug("Team ID required for get_channel_info")
            return None

        try:
            endpoint = f"/teams/{actual_team_id}/channels/{channel_id}"
            success, data, error = await self._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="get_channel_info",
            )

            if not success or not data:
                logger.debug(f"Failed to get channel info: {error}")
                return None

            return ChatChannel(
                id=channel_id,
                platform=self.platform_name,
                name=data.get("displayName"),
                is_private=data.get("membershipType") == "private",
                team_id=actual_team_id,
                metadata={
                    "description": data.get("description"),
                    "web_url": data.get("webUrl"),
                    "membership_type": data.get("membershipType"),
                },
            )

        except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, OSError) as e:
            logger.debug(f"Teams get_channel_info error: {e}")
            return None

    async def list_channels(
        self: _TeamsConnectorProtocol,
        team_id: str,
        include_private: bool = False,
        **kwargs: Any,
    ) -> list[ChatChannel]:
        """
        List all channels in a Microsoft Teams team.

        Uses Microsoft Graph API to enumerate channels.

        Args:
            team_id: Team ID to list channels for
            include_private: Whether to include private channels (default: False)
            **kwargs: Additional options

        Returns:
            List of ChatChannel objects
        """
        channels: list[ChatChannel] = []

        try:
            endpoint = f"/teams/{team_id}/channels"
            if not include_private:
                endpoint += "?$filter=membershipType eq 'standard'"

            success, data, error = await self._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="list_channels",
            )

            if not success or not data:
                logger.warning(f"Failed to list channels for team {team_id}: {error}")
                return channels

            channel_list = data.get("value", [])
            for channel_data in channel_list:
                channel = ChatChannel(
                    id=channel_data.get("id", ""),
                    platform=self.platform_name,
                    name=channel_data.get("displayName"),
                    is_private=channel_data.get("membershipType") == "private",
                    team_id=team_id,
                    metadata={
                        "description": channel_data.get("description"),
                        "web_url": channel_data.get("webUrl"),
                        "membership_type": channel_data.get("membershipType"),
                    },
                )
                channels.append(channel)

            logger.debug(f"Listed {len(channels)} channels for team {team_id}")
            return channels

        except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, OSError) as e:
            logger.error(f"Teams list_channels error: {e}")
            return channels

    async def get_user_info(
        self: _TeamsConnectorProtocol,
        user_id: str,
        **kwargs: Any,
    ) -> ChatUser | None:
        """
        Get information about a user via Microsoft Graph API.

        Args:
            user_id: User ID (AAD Object ID)
            **kwargs: Additional options

        Returns:
            ChatUser info or None
        """
        try:
            endpoint = f"/users/{user_id}"
            success, data, error = await self._graph_api_request(
                endpoint=endpoint,
                method="GET",
                operation="get_user_info",
            )

            if not success or not data:
                logger.debug(f"Failed to get user info: {error}")
                return None

            return ChatUser(
                id=user_id,
                platform=self.platform_name,
                username=data.get("userPrincipalName"),
                display_name=data.get("displayName"),
                email=data.get("mail"),
                metadata={
                    "job_title": data.get("jobTitle"),
                    "office_location": data.get("officeLocation"),
                    "department": data.get("department"),
                },
            )

        except (httpx.HTTPError, httpx.TimeoutException, RuntimeError, OSError) as e:
            logger.debug(f"Teams get_user_info error: {e}")
            return None
