"""
Global Knowledge Operations Mixin for Knowledge Mound.

Provides operations for global/public knowledge:
- store_verified_fact: Store verified facts in the global workspace
- query_global_knowledge: Query the global knowledge mound
- promote_to_global: Promote workspace knowledge to global
- get_system_facts: Get all system-verified facts

The global knowledge workspace (__system__) contains verified facts
that are accessible to all users regardless of their workspace.

NOTE: This is a mixin class designed to be composed with KnowledgeMound.
Attribute accesses like self._ensure_initialized, self.store, self.query, self.get, etc.
are provided by the composed class. A Protocol class (GlobalKnowledgeProtocol) is used
with cast() to provide type-safe access to these attributes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

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

    config: MoundConfig
    workspace_id: str
    _meta_store: Any | None
    _cache: Any | None
    _initialized: bool

    def _ensure_initialized(self) -> None: ...

    async def store(self, request: IngestionRequest) -> IngestionResult: ...

    async def query(
        self,
        query: str,
        sources: Any = ("all",),
        filters: Any = None,
        limit: int = 20,
        workspace_id: str | None = None,
    ) -> QueryResult: ...

    async def get(
        self, node_id: str, workspace_id: str | None = None
    ) -> KnowledgeItem | None: ...

    # Mixin methods that call each other
    async def store_verified_fact(
        self,
        content: str,
        source: str,
        confidence: float = 0.9,
        evidence_ids: list[str] | None = None,
        verified_by: str = "system",
        topics: list[str] | None = None,
    ) -> str: ...

    async def query_global_knowledge(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.5,
        topics: list[str] | None = None,
    ) -> list[KnowledgeItem]: ...


class GlobalKnowledgeMixin:
    """Mixin providing global knowledge operations for KnowledgeMound."""

    async def store_verified_fact(
        self,
        content: str,
        source: str,
        confidence: float = 0.9,
        evidence_ids: list[str] | None = None,
        verified_by: str | None = None,
        topics: list[str] | None = None,
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

        # Cast self to Protocol to access methods provided by composed KnowledgeMound class
        host = cast(GlobalKnowledgeProtocol, self)
        host._ensure_initialized()

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

        result = await host.store(request)
        logger.info("Stored verified fact %s in global knowledge", result.node_id)
        return result.node_id

    async def query_global_knowledge(
        self,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.0,
        topics: list[str] | None = None,
    ) -> list[KnowledgeItem]:
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

        # Cast self to Protocol to access methods provided by composed KnowledgeMound class
        host = cast(GlobalKnowledgeProtocol, self)
        host._ensure_initialized()

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

        result = await host.query(
            query=query,
            workspace_id=SYSTEM_WORKSPACE_ID,
            limit=limit,
            filters=filters,
        )

        return result.items

    async def promote_to_global(
        self,
        item_id: str,
        workspace_id: str,
        promoted_by: str,
        reason: str,
        additional_evidence: list[str] | None = None,
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
        # Cast self to Protocol to access methods provided by composed KnowledgeMound class
        host = cast(GlobalKnowledgeProtocol, self)
        host._ensure_initialized()

        # Get the original item
        item = await host.get(item_id, workspace_id=workspace_id)
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
        return await host.store_verified_fact(
            content=item.content,
            source=f"promoted_from:{workspace_id}:{item_id}",
            confidence=confidence_value,
            evidence_ids=all_evidence,
            verified_by=promoted_by,
            topics=(item.metadata or {}).get("topics", []),
        )

    async def get_system_facts(
        self,
        limit: int = 100,
        topics: list[str] | None = None,
    ) -> list[KnowledgeItem]:
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
        self,
        workspace_results: list[KnowledgeItem],
        query: str,
        global_limit: int = 5,
    ) -> list[KnowledgeItem]:
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
        # Cast self to Protocol to access methods provided by composed KnowledgeMound class
        host = cast(GlobalKnowledgeProtocol, self)
        host._ensure_initialized()

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

    def get_system_workspace_id(self) -> str:
        """Get the system workspace ID for global knowledge."""
        return SYSTEM_WORKSPACE_ID
