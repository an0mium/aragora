"""
ContinuumAdapter - Bridges ContinuumMemory to the Knowledge Mound.

This adapter enables the Knowledge Mound to query and store knowledge in
ContinuumMemory's multi-tier system while maintaining backward compatibility.

The adapter provides:
- Unified search interface (search_by_keyword)
- Bidirectional sync (store to both systems)
- Tier-to-importance mapping
- Cross-reference tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory, ContinuumMemoryEntry
    from aragora.knowledge.mound.types import KnowledgeItem, IngestionRequest

logger = logging.getLogger(__name__)


@dataclass
class ContinuumSearchResult:
    """Wrapper for continuum memory search results with adapter metadata."""

    entry: "ContinuumMemoryEntry"
    relevance_score: float = 0.0
    matched_keywords: List[str] = None

    def __post_init__(self) -> None:
        if self.matched_keywords is None:
            self.matched_keywords = []


class ContinuumAdapter:
    """
    Adapter that bridges ContinuumMemory to the Knowledge Mound.

    Provides methods that the Knowledge Mound expects for federated queries:
    - search_by_keyword: Text-based search across tiers
    - to_knowledge_item: Convert entries to unified format
    - sync_from_mound: Store mound items in continuum memory

    Usage:
        from aragora.memory.continuum import ContinuumMemory
        from aragora.knowledge.mound.adapters import ContinuumAdapter

        continuum = ContinuumMemory()
        adapter = ContinuumAdapter(continuum)

        # Search for memories
        results = adapter.search_by_keyword("type errors", limit=10)

        # Convert to knowledge items
        items = [adapter.to_knowledge_item(r) for r in results]
    """

    def __init__(
        self,
        continuum: "ContinuumMemory",
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            continuum: The ContinuumMemory instance to wrap
            enable_dual_write: If True, writes go to both systems during migration
        """
        self._continuum = continuum
        self._enable_dual_write = enable_dual_write

    @property
    def continuum(self) -> "ContinuumMemory":
        """Access the underlying ContinuumMemory."""
        return self._continuum

    def search_by_keyword(
        self,
        query: str,
        limit: int = 10,
        tiers: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> List["ContinuumMemoryEntry"]:
        """
        Search continuum memory by keyword query.

        This method wraps ContinuumMemory.retrieve() to provide the interface
        expected by KnowledgeMound._query_continuum().

        Args:
            query: Search query (keywords are OR'd)
            limit: Maximum results to return
            tiers: Optional list of tier names to filter (e.g., ["fast", "medium"])
            min_importance: Minimum importance threshold

        Returns:
            List of ContinuumMemoryEntry objects matching the query
        """
        from aragora.memory.tier_manager import MemoryTier

        # Convert tier names to MemoryTier enums
        tier_enums = None
        if tiers:
            tier_enums = []
            for tier_name in tiers:
                try:
                    tier_enums.append(MemoryTier(tier_name))
                except ValueError:
                    logger.warning(f"Unknown tier: {tier_name}, skipping")

        # Use ContinuumMemory's retrieve method
        entries = self._continuum.retrieve(
            query=query,
            tiers=tier_enums,
            limit=limit,
            min_importance=min_importance,
        )

        return list(entries)

    def get(self, entry_id: str) -> Optional["ContinuumMemoryEntry"]:
        """
        Get a specific entry by ID.

        Args:
            entry_id: The entry ID (may be prefixed with "cm_" from mound)

        Returns:
            ContinuumMemoryEntry or None
        """
        # Strip mound prefix if present
        if entry_id.startswith("cm_"):
            entry_id = entry_id[3:]

        return self._continuum.get(entry_id)

    async def get_async(self, entry_id: str) -> Optional["ContinuumMemoryEntry"]:
        """Async version of get for compatibility."""
        # Strip mound prefix if present
        if entry_id.startswith("cm_"):
            entry_id = entry_id[3:]

        return await self._continuum.get_async(entry_id)

    def to_knowledge_item(self, entry: "ContinuumMemoryEntry") -> "KnowledgeItem":
        """
        Convert a ContinuumMemoryEntry to a KnowledgeItem.

        Args:
            entry: The continuum memory entry

        Returns:
            KnowledgeItem for unified knowledge mound API
        """
        from aragora.knowledge.mound.types import (
            ConfidenceLevel,
            KnowledgeItem,
            KnowledgeSource,
        )

        # Map tier to confidence level
        tier_to_confidence = {
            "fast": ConfidenceLevel.LOW,  # Fast tier is volatile
            "medium": ConfidenceLevel.MEDIUM,
            "slow": ConfidenceLevel.HIGH,
            "glacial": ConfidenceLevel.VERIFIED,  # Glacial is most stable
        }
        confidence = tier_to_confidence.get(entry.tier.value, ConfidenceLevel.MEDIUM)

        # Build metadata
        metadata: Dict[str, Any] = {
            "tier": entry.tier.value,
            "surprise_score": entry.surprise_score,
            "consolidation_score": entry.consolidation_score,
            "update_count": entry.update_count,
            "success_rate": entry.success_rate,
        }
        if entry.red_line:
            metadata["red_line"] = True
            metadata["red_line_reason"] = entry.red_line_reason
        if entry.tags:
            metadata["tags"] = entry.tags
        if entry.cross_references:
            metadata["cross_references"] = entry.cross_references

        return KnowledgeItem(
            id=entry.knowledge_mound_id,  # Uses "cm_" prefix
            content=entry.content,
            source=KnowledgeSource.CONTINUUM,
            source_id=entry.id,
            confidence=confidence,
            created_at=datetime.fromisoformat(entry.created_at),
            updated_at=datetime.fromisoformat(entry.updated_at),
            metadata=metadata,
            importance=entry.importance,
        )

    def from_ingestion_request(
        self,
        request: "IngestionRequest",
        entry_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert an IngestionRequest to ContinuumMemory add() parameters.

        Args:
            request: The ingestion request from Knowledge Mound
            entry_id: Optional ID to use (generates one if not provided)

        Returns:
            Dict of parameters for ContinuumMemory.add()
        """
        import uuid
        from aragora.memory.tier_manager import MemoryTier

        # Map KnowledgeMound tier to ContinuumMemory tier
        tier_mapping = {
            1: MemoryTier.FAST,
            2: MemoryTier.MEDIUM,
            3: MemoryTier.SLOW,
            4: MemoryTier.GLACIAL,
        }
        tier = tier_mapping.get(request.tier, MemoryTier.SLOW)

        return {
            "id": entry_id or f"mound_{uuid.uuid4().hex[:12]}",
            "content": request.content,
            "tier": tier,
            "importance": request.confidence,
            "metadata": {
                "source_type": request.source_type.value,
                "debate_id": request.debate_id,
                "document_id": request.document_id,
                "agent_id": request.agent_id,
                "user_id": request.user_id,
                "workspace_id": request.workspace_id,
                "topics": request.topics,
                "mound_metadata": request.metadata,
            },
        }

    def store(
        self,
        content: str,
        importance: float = 0.5,
        tier: str = "slow",
        entry_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store content in continuum memory.

        Args:
            content: The content to store
            importance: Importance score (0-1)
            tier: Tier name ("fast", "medium", "slow", "glacial")
            entry_id: Optional ID (generated if not provided)
            metadata: Optional metadata dict

        Returns:
            The entry ID
        """
        import uuid
        from aragora.memory.tier_manager import MemoryTier

        if entry_id is None:
            entry_id = f"mound_{uuid.uuid4().hex[:12]}"

        tier_enum = MemoryTier(tier)

        self._continuum.add(
            id=entry_id,
            content=content,
            tier=tier_enum,
            importance=importance,
            metadata=metadata or {},
        )

        return entry_id

    def link_to_mound(
        self,
        entry_id: str,
        mound_node_id: str,
    ) -> None:
        """
        Link a continuum entry to a knowledge mound node.

        Creates a cross-reference from the continuum entry to the mound node,
        enabling bidirectional navigation.

        Args:
            entry_id: The continuum entry ID
            mound_node_id: The knowledge mound node ID
        """
        entry = self._continuum.get(entry_id)
        if entry:
            entry.add_cross_reference(mound_node_id)
            # Save the updated entry
            self._continuum.update(
                entry_id,
                metadata=entry.metadata,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the continuum memory."""
        return self._continuum.get_stats()

    def get_tier_metrics(self) -> Dict[str, Any]:
        """Get per-tier metrics."""
        return self._continuum.get_tier_metrics()
