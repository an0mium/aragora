"""
Knowledge + Chat Bridge Service.

Provides unified knowledge search and retrieval for chat platforms.
Bridges the gap between chat connectors (Slack, Teams, Discord, etc.) and
the Knowledge Mound for context-aware conversations and deliberations.

Key Features:
- Semantic search across Knowledge Mound from chat context
- Chat history to knowledge integration
- Real-time knowledge injection for conversations
- Multi-tenant workspace isolation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeSearchScope(str, Enum):
    """Scope for knowledge search."""

    ALL = "all"  # Search all accessible knowledge
    WORKSPACE = "workspace"  # Limit to workspace
    CHANNEL = "channel"  # Limit to channel-specific knowledge
    USER = "user"  # Limit to user's personal knowledge


class RelevanceStrategy(str, Enum):
    """Strategy for computing relevance."""

    SEMANTIC = "semantic"  # Embedding-based similarity
    KEYWORD = "keyword"  # Keyword/BM25 matching
    HYBRID = "hybrid"  # Combined semantic + keyword
    RECENCY = "recency"  # Favor recent knowledge


@dataclass
class KnowledgeSearchResult:
    """A single knowledge search result."""

    node_id: str
    content: str
    node_type: str
    confidence: float
    relevance_score: float
    source: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "content": self.content[:500] if len(self.content) > 500 else self.content,
            "node_type": self.node_type,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "provenance": self.provenance,
        }

    @property
    def combined_score(self) -> float:
        """Combined confidence and relevance score."""
        return (self.confidence * 0.4) + (self.relevance_score * 0.6)


@dataclass
class ChatKnowledgeContext:
    """Knowledge context for a chat conversation."""

    channel_id: str
    workspace_id: str
    query: str
    results: List[KnowledgeSearchResult]
    search_scope: KnowledgeSearchScope
    search_time_ms: float
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_id": self.channel_id,
            "workspace_id": self.workspace_id,
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "result_count": len(self.results),
            "search_scope": self.search_scope.value,
            "search_time_ms": self.search_time_ms,
            "suggestions": self.suggestions,
        }


class KnowledgeChatBridge:
    """
    Bridge service connecting chat platforms to the Knowledge Mound.

    Provides:
    - Semantic search from chat context
    - Knowledge injection for conversations
    - Chat history to knowledge conversion
    - Multi-tenant isolation
    """

    def __init__(
        self,
        mound=None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
        max_results: int = 10,
    ):
        """
        Initialize the Knowledge + Chat bridge.

        Args:
            mound: KnowledgeMound instance (lazy-loaded if None)
            enable_caching: Cache search results
            cache_ttl_seconds: Cache TTL in seconds
            max_results: Default max results per search
        """
        self._mound = mound
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_results = max_results
        self._cache: Dict[str, tuple] = {}  # query -> (results, timestamp)

    def _get_mound(self):
        """Lazy-load Knowledge Mound."""
        if self._mound is None:
            try:
                from aragora.knowledge.mound.facade import get_knowledge_mound

                self._mound = get_knowledge_mound()
            except ImportError:
                logger.warning("KnowledgeMound not available")
        return self._mound

    async def search_knowledge(
        self,
        query: str,
        workspace_id: str = "default",
        channel_id: Optional[str] = None,
        user_id: Optional[str] = None,
        scope: KnowledgeSearchScope = KnowledgeSearchScope.WORKSPACE,
        strategy: RelevanceStrategy = RelevanceStrategy.HYBRID,
        node_types: Optional[List[str]] = None,
        min_confidence: float = 0.3,
        max_results: Optional[int] = None,
    ) -> ChatKnowledgeContext:
        """
        Search knowledge relevant to a chat query.

        Args:
            query: Search query text
            workspace_id: Workspace for multi-tenant isolation
            channel_id: Optional channel context
            user_id: Optional user context
            scope: Search scope (all, workspace, channel, user)
            strategy: Relevance computation strategy
            node_types: Filter by node types (e.g., ["evidence", "pattern"])
            min_confidence: Minimum confidence threshold
            max_results: Max results to return

        Returns:
            ChatKnowledgeContext with search results
        """
        import time

        start_time = time.time()
        max_results = max_results or self.max_results

        # Check cache
        cache_key = f"{workspace_id}:{channel_id}:{query}:{scope.value}"
        if self.enable_caching and cache_key in self._cache:
            cached_results, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self.cache_ttl_seconds:
                return cached_results

        results: List[KnowledgeSearchResult] = []
        suggestions: List[str] = []

        mound = self._get_mound()
        if mound:
            try:
                # Build search parameters
                search_params: Dict[str, Any] = {
                    "query": query,
                    "limit": max_results * 2,  # Over-fetch for filtering
                    "min_confidence": min_confidence,
                }

                if scope == KnowledgeSearchScope.WORKSPACE:
                    search_params["workspace_id"] = workspace_id
                elif scope == KnowledgeSearchScope.CHANNEL and channel_id:
                    search_params["workspace_id"] = workspace_id
                    search_params["metadata_filter"] = {"channel_id": channel_id}
                elif scope == KnowledgeSearchScope.USER and user_id:
                    search_params["workspace_id"] = workspace_id
                    search_params["metadata_filter"] = {"user_id": user_id}

                if node_types:
                    search_params["node_types"] = node_types

                # Execute search
                if strategy == RelevanceStrategy.SEMANTIC:
                    raw_results = await mound.semantic_search(**search_params)
                elif strategy == RelevanceStrategy.KEYWORD:
                    raw_results = await mound.keyword_search(**search_params)
                else:  # HYBRID or RECENCY
                    raw_results = await mound.hybrid_search(**search_params)

                # Convert to search results
                for node, score in raw_results[:max_results]:
                    results.append(
                        KnowledgeSearchResult(
                            node_id=node.node_id,
                            content=node.content,
                            node_type=node.node_type,
                            confidence=node.confidence,
                            relevance_score=score,
                            source=node.source or "unknown",
                            created_at=node.created_at or datetime.now(timezone.utc),
                            metadata=node.metadata or {},
                            provenance=str(node.provenance) if node.provenance else None,
                        )
                    )

                # Generate suggestions for follow-up queries
                if len(results) >= 2:
                    suggestions = self._generate_suggestions(query, results)

            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")

        elapsed_ms = (time.time() - start_time) * 1000

        context = ChatKnowledgeContext(
            channel_id=channel_id or "",
            workspace_id=workspace_id,
            query=query,
            results=results,
            search_scope=scope,
            search_time_ms=elapsed_ms,
            suggestions=suggestions,
        )

        # Cache results
        if self.enable_caching:
            self._cache[cache_key] = (context, time.time())

        return context

    async def inject_knowledge_for_conversation(
        self,
        messages: List[Dict[str, Any]],
        workspace_id: str = "default",
        channel_id: Optional[str] = None,
        max_context_items: int = 5,
    ) -> List[KnowledgeSearchResult]:
        """
        Get relevant knowledge to inject into a conversation.

        Analyzes recent messages and finds relevant knowledge.

        Args:
            messages: Recent chat messages (newest first)
            workspace_id: Workspace for isolation
            channel_id: Channel context
            max_context_items: Max knowledge items to return

        Returns:
            List of relevant knowledge results
        """
        # Build query from recent messages
        query_parts = []
        for msg in messages[:5]:  # Use last 5 messages
            content = msg.get("content") or msg.get("text", "")
            if content:
                query_parts.append(content[:200])  # Truncate long messages

        if not query_parts:
            return []

        combined_query = " ".join(query_parts)

        # Search with conversation context
        context = await self.search_knowledge(
            query=combined_query,
            workspace_id=workspace_id,
            channel_id=channel_id,
            scope=KnowledgeSearchScope.WORKSPACE,
            strategy=RelevanceStrategy.HYBRID,
            max_results=max_context_items,
        )

        return context.results

    async def store_chat_as_knowledge(
        self,
        messages: List[Dict[str, Any]],
        workspace_id: str = "default",
        channel_id: str = "",
        channel_name: str = "",
        platform: str = "unknown",
        node_type: str = "chat_context",
        min_messages: int = 3,
    ) -> Optional[str]:
        """
        Store important chat messages as knowledge.

        Args:
            messages: Chat messages to store
            workspace_id: Workspace for isolation
            channel_id: Channel ID
            channel_name: Human-readable channel name
            platform: Chat platform (slack, teams, etc.)
            node_type: Type of knowledge node
            min_messages: Minimum messages required

        Returns:
            Node ID if stored, None otherwise
        """
        if len(messages) < min_messages:
            return None

        mound = self._get_mound()
        if not mound:
            return None

        try:
            # Build content from messages
            content_parts = []
            for msg in messages:
                author = msg.get("author") or msg.get("user", "Unknown")
                text = msg.get("content") or msg.get("text", "")
                if text:
                    content_parts.append(f"{author}: {text}")

            content = "\n".join(content_parts)

            # Create knowledge node
            from aragora.knowledge.mound_core import KnowledgeNode, ProvenanceChain

            node = KnowledgeNode(
                content=content[:5000],  # Limit content size
                node_type=node_type,
                source=f"{platform}:{channel_id}",
                confidence=0.7,
                metadata={
                    "workspace_id": workspace_id,
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "platform": platform,
                    "message_count": len(messages),
                },
                provenance=ProvenanceChain(
                    source=f"Chat conversation in {channel_name}",
                    transformations=["chat_to_knowledge"],
                ),
            )

            # Store in mound
            node_id = await mound.add_node(node, workspace_id=workspace_id)
            logger.info(f"Stored chat knowledge: {node_id} ({len(messages)} messages)")
            return node_id

        except Exception as e:
            logger.warning(f"Failed to store chat as knowledge: {e}")
            return None

    async def get_channel_knowledge_summary(
        self,
        channel_id: str,
        workspace_id: str = "default",
        max_items: int = 10,
    ) -> Dict[str, Any]:
        """
        Get a summary of knowledge related to a channel.

        Args:
            channel_id: Channel to summarize
            workspace_id: Workspace for isolation
            max_items: Max knowledge items

        Returns:
            Summary with stats and top knowledge items
        """
        context = await self.search_knowledge(
            query="*",  # Match all
            workspace_id=workspace_id,
            channel_id=channel_id,
            scope=KnowledgeSearchScope.CHANNEL,
            max_results=max_items,
        )

        # Compute stats
        node_types: Dict[str, int] = {}
        avg_confidence = 0.0

        for result in context.results:
            node_types[result.node_type] = node_types.get(result.node_type, 0) + 1
            avg_confidence += result.confidence

        if context.results:
            avg_confidence /= len(context.results)

        return {
            "channel_id": channel_id,
            "workspace_id": workspace_id,
            "total_items": len(context.results),
            "node_types": node_types,
            "avg_confidence": avg_confidence,
            "top_items": [r.to_dict() for r in context.results[:5]],
            "search_time_ms": context.search_time_ms,
        }

    def _generate_suggestions(
        self,
        query: str,
        results: List[KnowledgeSearchResult],
    ) -> List[str]:
        """Generate follow-up query suggestions."""
        suggestions = []

        # Extract unique node types
        node_types = set(r.node_type for r in results)
        for nt in list(node_types)[:2]:
            suggestions.append(f"More {nt} about: {query[:50]}")

        # Suggest related sources
        sources = set(r.source.split(":")[0] for r in results if ":" in r.source)
        for source in list(sources)[:1]:
            suggestions.append(f"From {source}: {query[:50]}")

        return suggestions[:3]

    def clear_cache(self, workspace_id: Optional[str] = None):
        """Clear the search cache."""
        if workspace_id:
            keys_to_remove = [k for k in self._cache if k.startswith(f"{workspace_id}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()


# Global instance
_bridge: Optional[KnowledgeChatBridge] = None


def get_knowledge_chat_bridge() -> KnowledgeChatBridge:
    """Get or create the global Knowledge + Chat bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = KnowledgeChatBridge()
    return _bridge


__all__ = [
    "KnowledgeChatBridge",
    "KnowledgeSearchScope",
    "RelevanceStrategy",
    "KnowledgeSearchResult",
    "ChatKnowledgeContext",
    "get_knowledge_chat_bridge",
]
