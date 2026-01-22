"""
Knowledge Mound Unified Store.

Provides a federated query interface across all Aragora knowledge systems:
- ContinuumMemory: Multi-tier temporal learning
- ConsensusMemory: Debate outcomes and agreements
- FactStore: Verified facts from document analysis
- WeaviateStore: Semantic embeddings for similarity search

The Knowledge Mound enables cross-system queries and knowledge linking,
implementing the "termite mound" architecture where all agents contribute
to and query from a shared knowledge superstructure.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from aragora.knowledge.unified.types import (
    ConfidenceLevel,
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    LinkResult,
    QueryFilters,
    QueryResult,
    RelationshipType,
    SourceFilter,
    StoreResult,
)

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.consensus import ConsensusMemory
    from aragora.knowledge.fact_store import FactStore
    from aragora.documents.indexing.weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeMoundConfig:
    """Configuration for the Knowledge Mound."""

    # Optional store references (will use defaults if not provided)
    continuum_memory: Optional["ContinuumMemory"] = None
    consensus_memory: Optional["ConsensusMemory"] = None
    fact_store: Optional["FactStore"] = None
    vector_store: Optional["WeaviateStore"] = None

    # Feature flags
    enable_cross_references: bool = True
    enable_vector_search: bool = True
    enable_link_inference: bool = False  # Experimental: auto-infer links

    # Query settings
    default_limit: int = 20
    max_limit: int = 100
    parallel_queries: bool = True  # Query sources in parallel

    # Cross-reference settings
    auto_link_threshold: float = 0.85  # Similarity threshold for auto-linking


class KnowledgeMound:
    """
    Unified knowledge retrieval and storage across all memory systems.

    The Knowledge Mound provides:
    1. Federated queries across ContinuumMemory, ConsensusMemory, FactStore, and VectorStore
    2. Cross-referencing between knowledge items
    3. Knowledge graph traversal via links
    4. Unified storage with automatic routing to appropriate stores

    Usage:
        mound = KnowledgeMound(config)
        await mound.initialize()

        # Query across all sources
        result = await mound.query("contract expiration dates")

        # Store new knowledge
        item_id = await mound.store(
            content="Contract expires 2025-12-31",
            source_type=KnowledgeSource.FACT,
            metadata={"document_id": "doc_123"},
        )

        # Link related items
        await mound.link(
            source_id="fact_123",
            target_id="consensus_456",
            relationship=RelationshipType.SUPPORTS,
        )
    """

    def __init__(self, config: Optional[KnowledgeMoundConfig] = None):
        self.config = config or KnowledgeMoundConfig()
        self._initialized = False

        # Store references (lazy-loaded)
        self._continuum: Optional["ContinuumMemory"] = config.continuum_memory if config else None
        self._consensus: Optional["ConsensusMemory"] = config.consensus_memory if config else None
        self._facts: Optional["FactStore"] = config.fact_store if config else None
        self._vectors: Optional["WeaviateStore"] = config.vector_store if config else None

        # In-memory link storage (will be persisted in Phase 1.2)
        self._links: Dict[str, KnowledgeLink] = {}
        self._source_links: Dict[str, List[str]] = {}  # source_id -> [link_ids]
        self._target_links: Dict[str, List[str]] = {}  # target_id -> [link_ids]

    async def initialize(self) -> None:
        """Initialize connections to all knowledge stores."""
        if self._initialized:
            return

        logger.info("Initializing Knowledge Mound...")

        # Initialize stores that weren't provided
        if self._continuum is None:
            try:
                from aragora.memory.continuum import ContinuumMemory

                self._continuum = ContinuumMemory()
                # ContinuumMemory initializes in __init__, no async initialize needed
            except ImportError:
                logger.warning("ContinuumMemory not available")

        if self._consensus is None:
            try:
                from aragora.memory.consensus import ConsensusMemory

                self._consensus = ConsensusMemory()
            except ImportError:
                logger.warning("ConsensusMemory not available")

        if self._facts is None:
            try:
                from aragora.knowledge.fact_store import FactStore

                self._facts = FactStore()
            except ImportError:
                logger.warning("FactStore not available")

        if self._vectors is None and self.config.enable_vector_search:
            try:
                from aragora.documents.indexing.weaviate_store import WeaviateStore

                self._vectors = WeaviateStore()
            except ImportError:
                logger.warning("WeaviateStore not available")

        self._initialized = True
        logger.info("Knowledge Mound initialized")

    async def query(
        self,
        query: str,
        sources: Sequence[SourceFilter] = ("all",),
        filters: Optional[QueryFilters] = None,
        limit: int = 20,
        include_links: bool = False,
    ) -> QueryResult:
        """
        Query across all configured knowledge stores.

        Args:
            query: Natural language query string
            sources: Which sources to query ("all" or specific sources)
            filters: Optional filters to apply
            limit: Maximum number of results
            include_links: Whether to include linked items in results

        Returns:
            QueryResult with items from all queried sources
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        limit = min(limit, self.config.max_limit)

        # Determine which sources to query
        source_list = self._resolve_sources(sources)

        # Query each source
        if self.config.parallel_queries:
            tasks = []
            for source in source_list:
                tasks.append(self._query_source(source, query, filters, limit))
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for source in source_list:
                try:
                    result = await self._query_source(source, query, filters, limit)
                    results.append(result)
                except Exception as e:
                    results.append(e)

        # Combine results
        all_items: List[KnowledgeItem] = []
        for i, result in enumerate(results):  # type: ignore[assignment]
            if isinstance(result, Exception):
                logger.warning(f"Query to {source_list[i]} failed: {result}")
            elif result:
                all_items.extend(result)

        # Sort by importance/relevance and limit
        all_items.sort(key=lambda x: x.importance or 0, reverse=True)
        all_items = all_items[:limit]

        # Optionally include linked items
        if include_links and all_items:
            linked_items = await self._get_linked_items([item.id for item in all_items])
            # Add linked items that aren't already in results
            existing_ids = {item.id for item in all_items}
            for linked in linked_items:
                if linked.id not in existing_ids:
                    all_items.append(linked)

        execution_time = (time.time() - start_time) * 1000

        return QueryResult(
            items=all_items,
            total_count=len(all_items),
            query=query,
            filters=filters,
            execution_time_ms=execution_time,
            sources_queried=source_list,
        )

    async def store(
        self,
        content: str,
        source_type: KnowledgeSource,
        metadata: Optional[Dict[str, Any]] = None,
        cross_references: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> StoreResult:
        """
        Store a new knowledge item with automatic routing to the appropriate store.

        Args:
            content: The knowledge content to store
            source_type: Which store to use
            metadata: Optional metadata to attach
            cross_references: Optional list of item IDs to link to
            importance: Importance score (0-1)

        Returns:
            StoreResult with the new item ID
        """
        if not self._initialized:
            await self.initialize()

        metadata = metadata or {}
        item_id = f"km_{uuid.uuid4().hex[:12]}"

        try:
            if source_type == KnowledgeSource.CONTINUUM:
                await self._store_to_continuum(item_id, content, metadata, importance)
            elif source_type == KnowledgeSource.FACT:
                await self._store_to_facts(item_id, content, metadata)
            elif source_type == KnowledgeSource.CONSENSUS:
                # Consensus entries are typically created by debates, not directly
                logger.warning("Direct storage to consensus memory not recommended")
                return StoreResult(
                    id=item_id,
                    source=source_type,
                    success=False,
                    message="Use debate system to create consensus entries",
                )
            else:
                return StoreResult(
                    id=item_id,
                    source=source_type,
                    success=False,
                    message=f"Storage not supported for source type: {source_type}",
                )

            # Create cross-references if enabled
            refs_created = 0
            if self.config.enable_cross_references and cross_references:
                for ref_id in cross_references:
                    result = await self.link(
                        source_id=item_id,
                        target_id=ref_id,
                        relationship=RelationshipType.RELATED_TO,
                    )
                    if result.success:
                        refs_created += 1

            return StoreResult(
                id=item_id,
                source=source_type,
                success=True,
                cross_references_created=refs_created,
            )

        except Exception as e:
            logger.error(f"Failed to store knowledge item: {e}")
            return StoreResult(
                id=item_id,
                source=source_type,
                success=False,
                message=str(e),
            )

    async def link(
        self,
        source_id: str,
        target_id: str,
        relationship: RelationshipType,
        confidence: float = 1.0,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LinkResult:
        """
        Create a link between two knowledge items.

        Args:
            source_id: ID of the source knowledge item
            target_id: ID of the target knowledge item
            relationship: Type of relationship
            confidence: Confidence in the relationship (0-1)
            created_by: Agent or user that created the link
            metadata: Optional metadata for the link

        Returns:
            LinkResult indicating success or failure
        """
        link_id = f"link_{uuid.uuid4().hex[:12]}"

        try:
            link = KnowledgeLink(
                id=link_id,
                source_id=source_id,
                target_id=target_id,
                relationship=relationship,
                confidence=confidence,
                created_at=datetime.now(timezone.utc),
                created_by=created_by,
                metadata=metadata or {},
            )

            # Store link
            self._links[link_id] = link

            # Update indexes
            if source_id not in self._source_links:
                self._source_links[source_id] = []
            self._source_links[source_id].append(link_id)

            if target_id not in self._target_links:
                self._target_links[target_id] = []
            self._target_links[target_id].append(link_id)

            logger.debug(f"Created link {link_id}: {source_id} -> {target_id} ({relationship})")

            return LinkResult(id=link_id, success=True)

        except Exception as e:
            logger.error(f"Failed to create link: {e}")
            return LinkResult(id=link_id, success=False, message=str(e))

    async def get_links(
        self,
        item_id: str,
        direction: str = "both",  # "outgoing", "incoming", "both"
        relationship: Optional[RelationshipType] = None,
    ) -> List[KnowledgeLink]:
        """
        Get all links for a knowledge item.

        Args:
            item_id: ID of the knowledge item
            direction: Which links to return
            relationship: Optional filter by relationship type

        Returns:
            List of matching links
        """
        links: List[KnowledgeLink] = []

        if direction in ("outgoing", "both"):
            for link_id in self._source_links.get(item_id, []):
                link = self._links.get(link_id)
                if link and (relationship is None or link.relationship == relationship):
                    links.append(link)

        if direction in ("incoming", "both"):
            for link_id in self._target_links.get(item_id, []):
                link = self._links.get(link_id)
                if link and (relationship is None or link.relationship == relationship):
                    links.append(link)

        return links

    async def get_graph(
        self,
        root_id: str,
        depth: int = 2,
        max_nodes: int = 50,
    ) -> Dict[str, Any]:
        """
        Get a knowledge subgraph starting from a root item.

        Args:
            root_id: ID of the root knowledge item
            depth: Maximum depth to traverse
            max_nodes: Maximum number of nodes to return

        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes: Dict[str, KnowledgeItem] = {}
        edges: List[Dict[str, Any]] = []
        visited: set = set()
        queue: List[tuple] = [(root_id, 0)]

        while queue and len(nodes) < max_nodes:
            current_id, current_depth = queue.pop(0)

            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)

            # Get the item
            item = await self._get_item_by_id(current_id)
            if item:
                nodes[current_id] = item

                # Get outgoing links
                if current_depth < depth:
                    links = await self.get_links(current_id, direction="outgoing")
                    for link in links:
                        edges.append(
                            {
                                "source": link.source_id,
                                "target": link.target_id,
                                "relationship": link.relationship.value,
                                "confidence": link.confidence,
                            }
                        )
                        if link.target_id not in visited:
                            queue.append((link.target_id, current_depth + 1))

        return {
            "nodes": [node.to_dict() for node in nodes.values()],
            "edges": edges,
            "root_id": root_id,
            "depth": depth,
        }

    # Private helper methods

    def _resolve_sources(self, sources: Sequence[SourceFilter]) -> List[KnowledgeSource]:
        """Resolve source filter to list of KnowledgeSource."""
        if "all" in sources:
            return [
                KnowledgeSource.CONTINUUM,
                KnowledgeSource.CONSENSUS,
                KnowledgeSource.FACT,
                KnowledgeSource.VECTOR,
            ]

        result = []
        for s in sources:
            if s != "all":
                result.append(KnowledgeSource(s))
        return result

    async def _query_source(
        self,
        source: KnowledgeSource,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
    ) -> List[KnowledgeItem]:
        """Query a specific knowledge source."""
        if source == KnowledgeSource.CONTINUUM:
            return await self._query_continuum(query, filters, limit)
        elif source == KnowledgeSource.CONSENSUS:
            return await self._query_consensus(query, filters, limit)
        elif source == KnowledgeSource.FACT:
            return await self._query_facts(query, filters, limit)
        elif source == KnowledgeSource.VECTOR:
            return await self._query_vectors(query, filters, limit)
        return []

    async def _query_continuum(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
    ) -> List[KnowledgeItem]:
        """Query ContinuumMemory."""
        if not self._continuum:
            return []

        try:
            # Use keyword matching for now (semantic search in Phase 1.2)
            entries = self._continuum.search_by_keyword(query, limit=limit)
            items = []
            for entry in entries:
                items.append(
                    KnowledgeItem(
                        id=f"cm_{entry.id}",
                        content=entry.content,
                        source=KnowledgeSource.CONTINUUM,
                        source_id=entry.id,
                        confidence=self._tier_to_confidence(entry.tier.value),
                        created_at=datetime.fromisoformat(entry.created_at),
                        updated_at=datetime.fromisoformat(entry.last_updated),
                        metadata={"tier": entry.tier.value, "tags": entry.tags},
                        importance=entry.importance,
                    )
                )
            return items
        except Exception as e:
            logger.error(f"Continuum query failed: {e}")
            return []

    async def _query_consensus(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
    ) -> List[KnowledgeItem]:
        """Query ConsensusMemory."""
        if not self._consensus:
            return []

        try:
            # Search by topic similarity
            entries = await self._consensus.search_by_topic(query, limit=limit)
            items = []
            for entry in entries:
                items.append(
                    KnowledgeItem(
                        id=f"cs_{entry.id}",
                        content=entry.final_claim or entry.topic,
                        source=KnowledgeSource.CONSENSUS,
                        source_id=entry.id,
                        confidence=self._strength_to_confidence(
                            entry.strength.value if hasattr(entry, "strength") else "moderate"
                        ),
                        created_at=(
                            datetime.fromisoformat(entry.created_at)
                            if hasattr(entry, "created_at")
                            else datetime.now(timezone.utc)
                        ),
                        updated_at=(
                            datetime.fromisoformat(entry.updated_at)
                            if hasattr(entry, "updated_at")
                            else datetime.now(timezone.utc)
                        ),
                        metadata={
                            "debate_id": entry.debate_id if hasattr(entry, "debate_id") else None,
                            "supporting_agents": (
                                entry.supporting_agents
                                if hasattr(entry, "supporting_agents")
                                else []
                            ),
                        },
                        importance=entry.confidence if hasattr(entry, "confidence") else 0.5,
                    )
                )
            return items
        except Exception as e:
            logger.error(f"Consensus query failed: {e}")
            return []

    async def _query_facts(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
    ) -> List[KnowledgeItem]:
        """Query FactStore."""
        if not self._facts:
            return []

        try:
            facts = await self._facts.query_facts(
                query=query,
                workspace_id=filters.workspace_id if filters else None,
                limit=limit,
            )
            items = []
            for fact in facts:
                items.append(
                    KnowledgeItem(
                        id=f"fc_{fact.id}",
                        content=fact.statement,
                        source=KnowledgeSource.FACT,
                        source_id=fact.id,
                        confidence=self._validation_to_confidence(fact.validation_status.value),
                        created_at=fact.created_at,
                        updated_at=fact.updated_at or fact.created_at,
                        metadata={
                            "evidence_ids": fact.evidence_ids,
                            "source_documents": fact.source_documents,
                            "tags": fact.tags,
                        },
                        importance=fact.confidence,
                    )
                )
            return items
        except Exception as e:
            logger.error(f"Facts query failed: {e}")
            return []

    async def _query_vectors(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
    ) -> List[KnowledgeItem]:
        """Query VectorStore for semantic similarity."""
        if not self._vectors:
            return []

        try:
            results = await self._vectors.search(
                query=query,
                limit=limit,
                filters=(
                    {"workspace_id": filters.workspace_id}
                    if filters and filters.workspace_id
                    else None
                ),
            )
            items = []
            for result in results:
                items.append(
                    KnowledgeItem(
                        id=f"vc_{result.id}",
                        content=result.content,
                        source=KnowledgeSource.VECTOR,
                        source_id=result.id,
                        confidence=ConfidenceLevel.MEDIUM,
                        created_at=datetime.now(
                            timezone.utc
                        ),  # Vector store may not have timestamps
                        updated_at=datetime.now(timezone.utc),
                        metadata=result.metadata or {},
                        importance=result.score if hasattr(result, "score") else 0.5,
                    )
                )
            return items
        except Exception as e:
            logger.error(f"Vector query failed: {e}")
            return []

    async def _store_to_continuum(
        self,
        item_id: str,
        content: str,
        metadata: Dict[str, Any],
        importance: float,
    ) -> None:
        """Store to ContinuumMemory."""
        if not self._continuum:
            raise RuntimeError("ContinuumMemory not available")

        await self._continuum.store(
            content=content,
            importance=importance,
            tags=metadata.get("tags", []),
            source_type=metadata.get("source_type", "knowledge_mound"),
        )

    async def _store_to_facts(
        self,
        item_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Store to FactStore."""
        if not self._facts:
            raise RuntimeError("FactStore not available")

        # add_fact is synchronous
        self._facts.add_fact(
            statement=content,
            evidence_ids=metadata.get("evidence_ids", []),
            source_documents=metadata.get("source_documents", []),
            workspace_id=metadata.get("workspace_id", "default"),
            confidence=metadata.get("confidence", 0.5),
            topics=metadata.get("topics", []),
        )

    async def _get_item_by_id(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by its Knowledge Mound ID."""
        # Parse the ID prefix to determine source
        if item_id.startswith("cm_"):
            return await self._get_continuum_item(item_id[3:])
        elif item_id.startswith("cs_"):
            return await self._get_consensus_item(item_id[3:])
        elif item_id.startswith("fc_"):
            return await self._get_fact_item(item_id[3:])
        elif item_id.startswith("vc_"):
            return await self._get_vector_item(item_id[3:])
        return None

    async def _get_continuum_item(self, source_id: str) -> Optional[KnowledgeItem]:
        """Get a ContinuumMemory item by source ID."""
        if not self._continuum:
            return None
        entry = self._continuum.get_entry(source_id)
        if entry:
            return KnowledgeItem(
                id=f"cm_{entry.id}",
                content=entry.content,
                source=KnowledgeSource.CONTINUUM,
                source_id=entry.id,
                confidence=self._tier_to_confidence(entry.tier.value),
                created_at=datetime.fromisoformat(entry.created_at),
                updated_at=datetime.fromisoformat(entry.last_updated),
                metadata={"tier": entry.tier.value},
                importance=entry.importance,
            )
        return None

    async def _get_consensus_item(self, source_id: str) -> Optional[KnowledgeItem]:
        """Get a ConsensusMemory item by source ID."""
        # Implementation depends on ConsensusMemory.get() method
        return None

    async def _get_fact_item(self, source_id: str) -> Optional[KnowledgeItem]:
        """Get a FactStore item by source ID."""
        if not self._facts:
            return None
        fact = await self._facts.get_fact(source_id)
        if fact:
            return KnowledgeItem(
                id=f"fc_{fact.id}",
                content=fact.statement,
                source=KnowledgeSource.FACT,
                source_id=fact.id,
                confidence=self._validation_to_confidence(fact.validation_status.value),
                created_at=fact.created_at,
                updated_at=fact.updated_at or fact.created_at,
                metadata={
                    "evidence_ids": fact.evidence_ids,
                    "source_documents": fact.source_documents,
                },
                importance=fact.confidence,
            )
        return None

    async def _get_vector_item(self, source_id: str) -> Optional[KnowledgeItem]:
        """Get a VectorStore item by source ID."""
        # Implementation depends on WeaviateStore.get() method
        return None

    async def _get_linked_items(self, item_ids: List[str]) -> List[KnowledgeItem]:
        """Get all items linked to the given item IDs."""
        linked_items = []
        seen_ids = set(item_ids)

        for item_id in item_ids:
            links = await self.get_links(item_id, direction="both")
            for link in links:
                target_id = link.target_id if link.source_id == item_id else link.source_id
                if target_id not in seen_ids:
                    seen_ids.add(target_id)
                    item = await self._get_item_by_id(target_id)
                    if item:
                        linked_items.append(item)

        return linked_items

    # Confidence level mapping helpers

    def _tier_to_confidence(self, tier: str) -> ConfidenceLevel:
        """Map ContinuumMemory tier to confidence level."""
        mapping = {
            "glacial": ConfidenceLevel.VERIFIED,
            "slow": ConfidenceLevel.HIGH,
            "medium": ConfidenceLevel.MEDIUM,
            "fast": ConfidenceLevel.LOW,
        }
        return mapping.get(tier, ConfidenceLevel.MEDIUM)

    def _strength_to_confidence(self, strength: str) -> ConfidenceLevel:
        """Map ConsensusMemory strength to confidence level."""
        mapping = {
            "unanimous": ConfidenceLevel.VERIFIED,
            "strong": ConfidenceLevel.HIGH,
            "moderate": ConfidenceLevel.MEDIUM,
            "weak": ConfidenceLevel.LOW,
            "split": ConfidenceLevel.LOW,
            "contested": ConfidenceLevel.UNVERIFIED,
        }
        return mapping.get(strength.lower(), ConfidenceLevel.MEDIUM)

    def _validation_to_confidence(self, status: str) -> ConfidenceLevel:
        """Map FactStore validation status to confidence level."""
        mapping = {
            "verified": ConfidenceLevel.VERIFIED,
            "unverified": ConfidenceLevel.MEDIUM,
            "contradicted": ConfidenceLevel.LOW,
        }
        return mapping.get(status.lower(), ConfidenceLevel.MEDIUM)
