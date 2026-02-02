"""
Evidence Collection Mixin for Chat Platform Connectors.

Contains methods for collecting chat messages as evidence for debates.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import ChatEvidence, ChatMessage

logger = logging.getLogger(__name__)


class EvidenceMixin:
    """
    Mixin providing evidence collection for chat connectors.

    Includes:
    - Evidence collection from channels
    - Message history retrieval
    - Relevance scoring
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform identifier."""
        raise NotImplementedError

    # ==========================================================================
    # Evidence Collection
    # ==========================================================================

    async def collect_evidence(
        self,
        channel_id: str,
        query: str | None = None,
        limit: int = 100,
        include_threads: bool = True,
        min_relevance: float = 0.0,
        **kwargs: Any,
    ) -> list["ChatEvidence"]:
        """
        Collect chat messages as evidence for debates.

        Retrieves messages from a channel and converts them to evidence format
        with provenance tracking and relevance scoring.

        Args:
            channel_id: Channel to collect evidence from
            query: Optional search query to filter messages
            limit: Maximum number of messages to retrieve
            include_threads: Whether to include threaded replies
            min_relevance: Minimum relevance score (0-1) for inclusion
            **kwargs: Platform-specific options

        Returns:
            List of ChatEvidence objects with source tracking
        """
        # Default implementation - subclasses should override for platform-specific APIs
        logger.debug(f"{self.platform_name} collect_evidence not fully implemented")
        return []

    async def get_channel_history(
        self,
        channel_id: str,
        limit: int = 100,
        oldest: str | None = None,
        latest: str | None = None,
        **kwargs: Any,
    ) -> list["ChatMessage"]:
        """
        Get message history from a channel.

        Args:
            channel_id: Channel to get history from
            limit: Maximum number of messages
            oldest: Start timestamp (platform-specific format)
            latest: End timestamp (platform-specific format)
            **kwargs: Platform-specific options

        Returns:
            List of ChatMessage objects
        """
        logger.debug(f"{self.platform_name} get_channel_history not implemented")
        return []

    def _message_matches_query(
        self,
        message: "ChatMessage",
        query: str,
    ) -> bool:
        """Check if a message matches the search query."""
        if not query:
            return True

        query_lower = query.lower()
        text_lower = (message.content or "").lower()

        # Simple keyword matching
        keywords = query_lower.split()
        return any(kw in text_lower for kw in keywords)

    def _compute_message_relevance(
        self,
        message: "ChatMessage",
        query: str | None = None,
    ) -> float:
        """Compute relevance score for a message."""
        if not query:
            return 1.0

        # Simple TF-based relevance
        query_lower = query.lower()
        text_lower = (message.content or "").lower()

        keywords = query_lower.split()
        if not keywords or not text_lower:
            return 0.0

        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords)


__all__ = ["EvidenceMixin"]
