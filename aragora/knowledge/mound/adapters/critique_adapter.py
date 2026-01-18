"""
CritiqueAdapter - Bridges CritiqueStore to the Knowledge Mound.

This adapter enables the Knowledge Mound to query successful critique patterns
and agent reputation data from the CritiqueStore.

The adapter provides:
- Pattern search by issue type
- Agent reputation integration
- Pattern-to-KnowledgeItem conversion
- Surprise-based learning integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.memory.store import CritiqueStore, Pattern, AgentReputation
    from aragora.knowledge.mound.types import KnowledgeItem

logger = logging.getLogger(__name__)


@dataclass
class CritiqueSearchResult:
    """Wrapper for critique pattern search results."""

    pattern: "Pattern"
    relevance_score: float = 0.0
    matched_category: bool = False


class CritiqueAdapter:
    """
    Adapter that bridges CritiqueStore to the Knowledge Mound.

    Provides methods that the Knowledge Mound expects for federated queries:
    - search_patterns: Find relevant critique patterns
    - to_knowledge_item: Convert patterns to unified format
    - get_agent_insights: Agent reputation for cultural learning

    Usage:
        from aragora.memory.store import CritiqueStore
        from aragora.knowledge.mound.adapters import CritiqueAdapter

        store = CritiqueStore()
        adapter = CritiqueAdapter(store)

        # Search for patterns
        results = adapter.search_patterns("performance", limit=10)

        # Convert to knowledge items
        items = [adapter.to_knowledge_item(r) for r in results]
    """

    def __init__(
        self,
        store: "CritiqueStore",
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            store: The CritiqueStore instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._store = store
        self._enable_dual_write = enable_dual_write

    @property
    def store(self) -> "CritiqueStore":
        """Access the underlying CritiqueStore."""
        return self._store

    def search_patterns(
        self,
        query: str,
        limit: int = 10,
        issue_type: Optional[str] = None,
        min_success: int = 1,
    ) -> List["Pattern"]:
        """
        Search critique patterns by query.

        This method wraps CritiqueStore.retrieve_patterns() to provide
        the interface expected by KnowledgeMound._query_critique().

        If query matches a known issue type, filters by that type.
        Otherwise, retrieves patterns and filters by keyword match.

        Args:
            query: Search query
            limit: Maximum results to return
            issue_type: Optional explicit issue type filter
            min_success: Minimum success count for patterns

        Returns:
            List of Pattern objects matching the query
        """
        # Known issue types from CritiqueStore._categorize_issue()
        known_types = {
            "performance",
            "security",
            "correctness",
            "clarity",
            "architecture",
            "completeness",
            "testing",
            "general",
        }

        # Check if query is an issue type
        query_lower = query.lower().strip()
        detected_type = None

        if issue_type:
            detected_type = issue_type
        elif query_lower in known_types:
            detected_type = query_lower
        else:
            # Check for type keywords in query
            for t in known_types:
                if t in query_lower:
                    detected_type = t
                    break

        # Retrieve patterns
        patterns = self._store.retrieve_patterns(
            issue_type=detected_type,
            min_success=min_success,
            limit=limit * 2 if detected_type is None else limit,  # Get more if filtering
        )

        # If no type filter, also filter by keyword in issue/suggestion text
        if detected_type is None and patterns:
            query_words = set(query_lower.split())
            filtered = []
            for pattern in patterns:
                pattern_words = set(pattern.issue_text.lower().split())
                if pattern.suggestion_text:
                    pattern_words.update(pattern.suggestion_text.lower().split())
                if query_words & pattern_words:  # Any word overlap
                    filtered.append(pattern)
            patterns = filtered[:limit]

        return patterns

    def get(self, pattern_id: str) -> Optional["Pattern"]:
        """
        Get a specific pattern by ID.

        Args:
            pattern_id: The pattern ID (may be prefixed with "cr_" from mound)

        Returns:
            Pattern or None
        """
        # Strip mound prefix if present
        if pattern_id.startswith("cr_"):
            pattern_id = pattern_id[3:]

        # CritiqueStore doesn't have a direct get_pattern method
        # We retrieve all patterns for the issue type and filter
        # This is inefficient but works for now
        patterns = self._store.retrieve_patterns(min_success=0, limit=1000)
        for pattern in patterns:
            if pattern.id == pattern_id:
                return pattern
        return None

    async def get_async(self, pattern_id: str) -> Optional["Pattern"]:
        """Async version of get for compatibility."""
        return self.get(pattern_id)

    def to_knowledge_item(
        self,
        pattern: "Pattern",
    ) -> "KnowledgeItem":
        """
        Convert a Pattern to a KnowledgeItem.

        Args:
            pattern: The critique pattern

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map success rate to confidence
        success_rate = pattern.success_rate
        if success_rate >= 0.8:
            confidence = ConfidenceLevel.HIGH
        elif success_rate >= 0.6:
            confidence = ConfidenceLevel.MEDIUM
        elif success_rate >= 0.4:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.UNCERTAIN

        # Build content from issue + suggestion
        content = pattern.issue_text
        if pattern.suggestion_text:
            content += f"\n\nSuggestion: {pattern.suggestion_text}"

        # Build metadata
        metadata: Dict[str, Any] = {
            "issue_type": pattern.issue_type,
            "success_count": pattern.success_count,
            "failure_count": pattern.failure_count,
            "success_rate": success_rate,
            "avg_severity": pattern.avg_severity,
            "example_task": pattern.example_task,
        }

        return KnowledgeItem(
            id=f"cr_{pattern.id}",
            content=content,
            source=KnowledgeSource.CRITIQUE,
            source_id=pattern.id,
            confidence=confidence,
            created_at=datetime.fromisoformat(pattern.created_at),
            updated_at=datetime.fromisoformat(pattern.updated_at),
            metadata=metadata,
            importance=success_rate,
        )

    def get_agent_reputation(
        self,
        agent_name: str,
    ) -> Optional["AgentReputation"]:
        """
        Get reputation data for an agent.

        This is useful for culture accumulation - agents with high reputation
        contribute more weight to organizational patterns.

        Args:
            agent_name: The agent name

        Returns:
            AgentReputation or None
        """
        return self._store.get_reputation(agent_name)

    def get_top_agents(
        self,
        limit: int = 10,
    ) -> List["AgentReputation"]:
        """
        Get top agents by reputation.

        Args:
            limit: Maximum agents to return

        Returns:
            List of AgentReputation sorted by score
        """
        reputations = self._store.get_all_reputations(limit=limit)
        # Sort by reputation score
        return sorted(reputations, key=lambda r: r.reputation_score, reverse=True)

    def get_agent_vote_weights(
        self,
        agent_names: List[str],
    ) -> Dict[str, float]:
        """
        Get vote weights for multiple agents.

        Vote weights are based on historical performance and can be used
        for weighted consensus or culture pattern strength.

        Args:
            agent_names: List of agent names

        Returns:
            Dict mapping agent names to vote weights (0.4-1.6 range)
        """
        return self._store.get_vote_weights_batch(agent_names)

    def get_patterns_by_type(
        self,
        issue_type: str,
        limit: int = 20,
    ) -> List["Pattern"]:
        """
        Get patterns for a specific issue type.

        Args:
            issue_type: The issue type (performance, security, etc.)
            limit: Maximum patterns to return

        Returns:
            List of Pattern objects
        """
        return self._store.retrieve_patterns(
            issue_type=issue_type,
            min_success=1,
            limit=limit,
        )

    def get_surprising_patterns(
        self,
        limit: int = 10,
    ) -> List["Pattern"]:
        """
        Get patterns with high surprise scores.

        High surprise patterns are particularly valuable as they represent
        unexpected successes that deviate from base rates.

        Args:
            limit: Maximum patterns to return

        Returns:
            List of Pattern objects sorted by surprise
        """
        # Retrieve more patterns and filter/sort by surprise
        patterns = self._store.retrieve_patterns(min_success=1, limit=limit * 3)

        # The retrieve_patterns method already incorporates surprise in its ranking
        # but we can re-sort to prioritize pure surprise
        return patterns[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the critique store."""
        return self._store.get_stats()

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about archived patterns."""
        return self._store.get_archive_stats()
