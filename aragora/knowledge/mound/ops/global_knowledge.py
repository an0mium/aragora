"""
Global Knowledge Operations Mixin for Knowledge Mound.

Provides operations for global/public knowledge:
- store_verified_fact: Store verified facts in the global workspace
- query_global_knowledge: Query the global knowledge mound
- promote_to_global: Promote workspace knowledge to global
- get_system_facts: Get all system-verified facts

The global knowledge workspace (__system__) contains verified facts
that are accessible to all users regardless of their workspace.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        IngestionRequest,
        IngestionResult,
        KnowledgeItem,
        MoundConfig,
        QueryResult,
    )

logger = logging.getLogger(__name__)

# Special workspace ID for global/system knowledge
SYSTEM_WORKSPACE_ID = "__system__"


class GlobalKnowledgeProtocol(Protocol):
    """Protocol defining expected interface for GlobalKnowledge mixin."""

    config: "MoundConfig"
    workspace_id: str
    _meta_store: Optional[Any]
    _cache: Optional[Any]
    _initialized: bool

    def _ensure_initialized(self) -> None: ...

    async def store(self, request: "IngestionRequest") -> "IngestionResult": ...

    async def query(
        self,
        query: str,
        sources: Any = ("all",),
        filters: Any = None,
        limit: int = 20,
        workspace_id: Optional[str] = None,
    ) -> "QueryResult": ...

    async def get(
        self, node_id: str, workspace_id: Optional[str] = None
    ) -> Optional["KnowledgeItem"]: ...

    # Mixin methods that call each other
    async def store_verified_fact(
        self,
        content: str,
        source: str,
        confidence: float = 0.9,
        evidence_ids: Optional[List[str]] = None,
        verified_by: str = "system",
        topics: Optional[List[str]] = None,
    ) -> str: ...

    async def query_global_knowledge(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.5,
        topics: Optional[List[str]] = None,
    ) -> List["KnowledgeItem"]: ...


class GlobalKnowledgeMixin:
    """Mixin providing global knowledge operations for KnowledgeMound."""

    async def store_verified_fact(
        self: GlobalKnowledgeProtocol,
        content: str,
        source: str,
        confidence: float = 0.9,
        evidence_ids: Optional[List[str]] = None,
        verified_by: Optional[str] = None,
        topics: Optional[List[str]] = None,
    ) -> str:
        """
        Store a verified fact in the global knowledge mound.

        Verified facts are accessible to ALL users across all workspaces.
        Only trusted sources should store verified facts.

        Args:
            content: The fact content
            source: Source identifier (e.g., "consensus:debate_123", "admin:manual")
            confidence: Confidence score (0.0-1.0)
            evidence_ids: IDs of supporting evidence
            verified_by: User/system that verified this fact
            topics: Topic tags for the fact

        Returns:
            The node ID of the stored fact
        """
        from aragora.knowledge.mound.types import (
            IngestionRequest,
            KnowledgeSource,
            VisibilityLevel,
        )

        self._ensure_initialized()

        request = IngestionRequest(
            content=content,
            workspace_id=SYSTEM_WORKSPACE_ID,
            source_type=KnowledgeSource.FACT,
            confidence=confidence,
            node_type="fact",
            tier="glacial",  # Long-lived facts
            topics=topics or [],
            metadata={
                "source": source,
                "verified_by": verified_by,
                "verified_at": datetime.now().isoformat(),
                "evidence_ids": evidence_ids or [],
                "visibility": VisibilityLevel.SYSTEM.value,
                "is_global": True,
            },
        )

        result = await self.store(request)
        logger.info(f"Stored verified fact {result.node_id} in global knowledge")
        return result.node_id

    async def query_global_knowledge(
        self: GlobalKnowledgeProtocol,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.0,
        topics: Optional[List[str]] = None,
    ) -> List["KnowledgeItem"]:
        """
        Query the global knowledge mound (verified facts only).

        Global knowledge is accessible to all users without workspace restrictions.

        Args:
            query: Search query
            limit: Maximum number of results
            min_confidence: Minimum confidence threshold
            topics: Filter by topics

        Returns:
            List of matching knowledge items from the global workspace
        """
        from aragora.knowledge.mound.types import ConfidenceLevel, QueryFilters

        self._ensure_initialized()

        filters = None
        if min_confidence > 0 or topics:
            # Map float confidence to ConfidenceLevel enum
            confidence_level = None
            if min_confidence > 0:
                if min_confidence >= 0.9:
                    confidence_level = ConfidenceLevel.VERIFIED
                elif min_confidence >= 0.7:
                    confidence_level = ConfidenceLevel.HIGH
                elif min_confidence >= 0.5:
                    confidence_level = ConfidenceLevel.MEDIUM
                else:
                    confidence_level = ConfidenceLevel.LOW

            filters = QueryFilters(
                min_confidence=confidence_level,
                tags=topics,  # QueryFilters uses 'tags', not 'topics'
            )

        result = await self.query(
            query=query,
            workspace_id=SYSTEM_WORKSPACE_ID,
            limit=limit,
            filters=filters,
        )

        return result.items

    async def promote_to_global(
        self: GlobalKnowledgeProtocol,
        item_id: str,
        workspace_id: str,
        promoted_by: str,
        reason: str,
        additional_evidence: Optional[List[str]] = None,
    ) -> str:
        """
        Promote a workspace knowledge item to global verified fact.

        This copies the item to the system workspace with SYSTEM visibility,
        creating a new global fact that references the original.

        Args:
            item_id: ID of the item to promote
            workspace_id: Workspace containing the original item
            promoted_by: User/system promoting the item
            reason: Reason for promotion (e.g., "high_consensus", "verified_external")
            additional_evidence: Additional evidence IDs supporting promotion

        Returns:
            The node ID of the new global fact

        Raises:
            ValueError: If the original item is not found
        """
        self._ensure_initialized()

        # Get the original item
        item = await self.get(item_id, workspace_id=workspace_id)
        if not item:
            raise ValueError(f"Item {item_id} not found in workspace {workspace_id}")

        # Extract confidence - handle both enum and float
        raw_confidence = item.confidence
        confidence_value: float = 0.8
        if hasattr(raw_confidence, "value"):
            # It's an enum like ConfidenceLevel
            confidence_map = {
                "verified": 0.95,
                "high": 0.85,
                "medium": 0.7,
                "low": 0.5,
                "speculative": 0.3,
            }
            confidence_value = confidence_map.get(raw_confidence.value, 0.7)
        elif isinstance(raw_confidence, (int, float)):
            confidence_value = float(raw_confidence)

        # Get existing evidence from metadata
        existing_evidence = (item.metadata or {}).get("evidence_ids", [])
        all_evidence = existing_evidence + (additional_evidence or [])

        # Store as global fact
        return await self.store_verified_fact(
            content=item.content,
            source=f"promoted_from:{workspace_id}:{item_id}",
            confidence=confidence_value,
            evidence_ids=all_evidence,
            verified_by=promoted_by,
            topics=(item.metadata or {}).get("topics", []),
        )

    async def get_system_facts(
        self: GlobalKnowledgeProtocol,
        limit: int = 100,
        topics: Optional[List[str]] = None,
    ) -> List["KnowledgeItem"]:
        """
        Get all verified facts from the global knowledge mound.

        Args:
            limit: Maximum number of facts to return
            topics: Filter by specific topics

        Returns:
            List of system-verified facts
        """
        # Query with empty string to get all items
        return await self.query_global_knowledge(
            query="",
            limit=limit,
            topics=topics,
        )

    async def merge_global_results(
        self: GlobalKnowledgeProtocol,
        workspace_results: List["KnowledgeItem"],
        query: str,
        global_limit: int = 5,
    ) -> List["KnowledgeItem"]:
        """
        Merge workspace query results with relevant global knowledge.

        This is useful for enriching workspace-specific queries with
        authoritative global facts.

        Args:
            workspace_results: Results from workspace query
            query: Original query string
            global_limit: Maximum global items to add

        Returns:
            Merged list with workspace results + relevant global facts
        """
        self._ensure_initialized()

        # Get global results
        global_results = await self.query_global_knowledge(
            query=query,
            limit=global_limit,
        )

        # Deduplicate by content hash if available
        seen_hashes = set()
        for item in workspace_results:
            content_hash = (item.metadata or {}).get("content_hash", item.content[:100])
            seen_hashes.add(content_hash)

        # Add global items that aren't duplicates
        merged = list(workspace_results)
        for item in global_results:
            content_hash = (item.metadata or {}).get("content_hash", item.content[:100])
            if content_hash not in seen_hashes:
                merged.append(item)
                seen_hashes.add(content_hash)

        # Re-sort by importance
        merged.sort(key=lambda x: x.importance or 0, reverse=True)

        return merged

    def get_system_workspace_id(self: GlobalKnowledgeProtocol) -> str:
        """Get the system workspace ID for global knowledge."""
        return SYSTEM_WORKSPACE_ID
