"""
Rich Context Mixin for Chat Platform Connectors.

Contains methods for fetching and formatting rich context for deliberation.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import ChannelContext, ChatMessage

logger = logging.getLogger(__name__)


class RichContextMixin:
    """
    Mixin providing rich context fetching for chat connectors.

    Includes:
    - Basic context fetching
    - Rich context with topics, sentiment, activity patterns
    - LLM-ready context formatting
    """

    # These methods are expected from the base class/other mixins
    get_channel_info: Any
    get_channel_history: Any

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier."""
        ...

    # ==========================================================================
    # Channel Context for Deliberation
    # ==========================================================================

    async def fetch_context(
        self,
        channel_id: str,
        lookback_minutes: int = 60,
        max_messages: int = 50,
        include_participants: bool = True,
        thread_id: str | None = None,
        **kwargs: Any,
    ) -> ChannelContext:
        """
        Fetch recent context from a channel for deliberation.

        This method retrieves recent messages and participants from a channel
        to provide context for multi-agent vetted decisionmaking sessions. It's used by the
        orchestration handler to auto-fetch context before starting debates.

        Args:
            channel_id: Channel to fetch context from
            lookback_minutes: How far back to look for messages (default: 60)
            max_messages: Maximum messages to retrieve (default: 50)
            include_participants: Whether to extract participant info (default: True)
            thread_id: Optional thread/conversation to focus on
            **kwargs: Platform-specific options

        Returns:
            ChannelContext with messages and metadata

        Example:
            context = await slack.fetch_context("C123456", lookback_minutes=30)
            deliberation_context = context.to_context_string()
        """
        from datetime import datetime, timedelta, timezone

        from ..models import ChannelContext, ChatChannel, ChatUser

        warnings = []

        # Get channel info
        channel = await self.get_channel_info(channel_id)
        if not channel:
            # Create a basic channel object if lookup failed
            channel = ChatChannel(
                id=channel_id,
                platform=self.platform_name,
            )
            warnings.append(f"Could not fetch channel info for {channel_id}")

        # Calculate timestamp for lookback
        oldest_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)

        # Platform-specific oldest timestamp conversion
        oldest_str = self._format_timestamp_for_api(oldest_time)

        # Fetch messages
        messages = await self.get_channel_history(
            channel_id=channel_id,
            limit=max_messages,
            oldest=oldest_str,
            **kwargs,
        )

        # Extract participants
        participants: list[ChatUser] = []
        if include_participants and messages:
            seen_users: dict[str, ChatUser] = {}
            for msg in messages:
                if msg.author.id not in seen_users:
                    seen_users[msg.author.id] = msg.author
            participants = list(seen_users.values())

        # Calculate timestamps
        oldest_timestamp = None
        newest_timestamp = None
        if messages:
            oldest_timestamp = min(m.timestamp for m in messages)
            newest_timestamp = max(m.timestamp for m in messages)

        context = ChannelContext(
            channel=channel,
            messages=messages,
            participants=participants,
            oldest_timestamp=oldest_timestamp,
            newest_timestamp=newest_timestamp,
            message_count=len(messages),
            participant_count=len(participants),
            warnings=warnings,
            metadata={
                "lookback_minutes": lookback_minutes,
                "max_messages": max_messages,
                "thread_id": thread_id,
            },
        )

        logger.debug(
            "Fetched context from %s channel %s: %s messages, %s participants", self.platform_name, channel_id, len(messages), len(participants)
        )

        return context

    def _format_timestamp_for_api(self, timestamp: Any) -> str | None:
        """
        Format a datetime for the platform's API.

        Override in subclasses for platform-specific formatting.
        Default returns ISO format.
        """
        from datetime import datetime

        if isinstance(timestamp, datetime):
            return timestamp.isoformat()
        return str(timestamp) if timestamp else None

    # ==========================================================================
    # Rich Context Injection (ClawdBot Pattern)
    # ==========================================================================

    async def fetch_rich_context(
        self,
        channel_id: str,
        lookback_minutes: int = 60,
        max_messages: int = 50,
        include_participants: bool = True,
        include_topics: bool = True,
        include_sentiment: bool = False,
        thread_id: str | None = None,
        format_for_llm: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Fetch rich context from a channel for LLM prompt enrichment.

        This implements the ClawdBot pattern of context injection, providing
        comprehensive channel state for multi-agent deliberation. Returns
        structured context suitable for injection into LLM prompts.

        Args:
            channel_id: Channel to fetch context from
            lookback_minutes: How far back to look for messages (default: 60)
            max_messages: Maximum messages to retrieve (default: 50)
            include_participants: Whether to extract participant info (default: True)
            include_topics: Whether to extract discussion topics (default: True)
            include_sentiment: Whether to analyze sentiment (default: False)
            thread_id: Optional thread/conversation to focus on
            format_for_llm: Whether to include LLM-ready formatted string (default: True)
            **kwargs: Platform-specific options

        Returns:
            Dict containing:
                - channel: Channel information
                - messages: List of recent messages
                - participants: List of active participants
                - topics: Extracted discussion topics (if include_topics)
                - sentiment: Sentiment analysis (if include_sentiment)
                - statistics: Message statistics (counts, activity patterns)
                - formatted_context: LLM-ready context string (if format_for_llm)
                - metadata: Additional context metadata

        Example:
            context = await connector.fetch_rich_context(
                "C123456",
                lookback_minutes=30,
                include_topics=True,
            )

            # Use formatted context in LLM prompt
            prompt = f\"\"\"
            Channel context:
            {context['formatted_context']}

            Based on this discussion, respond to: {user_query}
            \"\"\"
        """
        from datetime import datetime, timezone

        from ..rich_context import (
            analyze_sentiment as _analyze_sentiment_impl,
            calculate_activity_patterns as _calculate_activity_patterns_impl,
            extract_topics as _extract_topics_impl,
            format_context_for_llm as _format_context_for_llm_impl,
        )

        # Fetch base context
        base_context = await self.fetch_context(
            channel_id=channel_id,
            lookback_minutes=lookback_minutes,
            max_messages=max_messages,
            include_participants=include_participants,
            thread_id=thread_id,
            **kwargs,
        )

        # Build rich context
        rich_context: dict[str, Any] = {
            "channel": {
                "id": base_context.channel.id,
                "name": base_context.channel.name,
                "platform": base_context.channel.platform,
                "type": base_context.channel.channel_type,
            },
            "messages": [
                {
                    "id": msg.id,
                    "author": msg.author.display_name or msg.author.username or msg.author.id,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "thread_id": msg.thread_id,
                }
                for msg in base_context.messages
            ],
            "participants": [
                {
                    "id": p.id,
                    "name": p.display_name or p.username or p.id,
                    "is_bot": p.is_bot,
                }
                for p in base_context.participants
            ],
            "statistics": {
                "message_count": base_context.message_count,
                "participant_count": base_context.participant_count,
                "timespan_minutes": lookback_minutes,
                "oldest_message": (
                    base_context.oldest_timestamp.isoformat()
                    if base_context.oldest_timestamp
                    else None
                ),
                "newest_message": (
                    base_context.newest_timestamp.isoformat()
                    if base_context.newest_timestamp
                    else None
                ),
            },
            "metadata": {
                "platform": self.platform_name,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "thread_id": thread_id,
                **base_context.metadata,
            },
            "warnings": base_context.warnings,
        }

        # Extract discussion topics
        if include_topics:
            topics = _extract_topics_impl(base_context.messages)
            rich_context["topics"] = topics

        # Analyze sentiment (basic implementation)
        if include_sentiment:
            sentiment = _analyze_sentiment_impl(base_context.messages)
            rich_context["sentiment"] = sentiment

        # Calculate activity patterns
        rich_context["activity"] = _calculate_activity_patterns_impl(base_context.messages)

        # Format for LLM consumption
        if format_for_llm:
            rich_context["formatted_context"] = _format_context_for_llm_impl(rich_context)

        logger.debug(
            "Fetched rich context from %s channel %s: %s messages, %s topics extracted", self.platform_name, channel_id, len(base_context.messages), len(rich_context.get('topics', []))
        )

        return rich_context

    def _extract_topics(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """
        Extract discussion topics from messages.

        Simple keyword extraction - can be overridden for more sophisticated
        NLP-based topic extraction.

        Args:
            messages: List of messages to analyze

        Returns:
            List of topic dicts with topic and frequency
        """
        from ..rich_context import extract_topics as _extract_topics_impl

        return _extract_topics_impl(messages)

    def _analyze_sentiment(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """
        Basic sentiment analysis of messages.

        Simple keyword-based sentiment - can be overridden for more sophisticated
        analysis using NLP models.

        Args:
            messages: List of messages to analyze

        Returns:
            Dict with sentiment metrics
        """
        from ..rich_context import analyze_sentiment as _analyze_sentiment_impl

        return _analyze_sentiment_impl(messages)

    def _calculate_activity_patterns(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """
        Calculate activity patterns from messages.

        Args:
            messages: List of messages to analyze

        Returns:
            Dict with activity metrics
        """
        from ..rich_context import calculate_activity_patterns as _calculate_activity_patterns_impl

        return _calculate_activity_patterns_impl(messages)

    def _format_context_for_llm(self, rich_context: dict[str, Any]) -> str:
        """
        Format rich context into an LLM-ready string.

        Args:
            rich_context: The rich context dictionary

        Returns:
            Formatted string suitable for LLM prompt injection
        """
        from ..rich_context import format_context_for_llm as _format_context_for_llm_impl

        return _format_context_for_llm_impl(rich_context)


__all__ = ["RichContextMixin"]
