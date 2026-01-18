"""
Knowledge Mound Facade - Unified knowledge storage with production backends.

This is the main entry point for the enhanced Knowledge Mound system,
providing a unified API over multiple storage backends (SQLite, PostgreSQL, Redis)
and integrating staleness detection and culture accumulation.

Usage:
    from aragora.knowledge.mound import KnowledgeMound, MoundConfig

    config = MoundConfig(
        backend=MoundBackend.POSTGRES,
        postgres_url="postgresql://user:pass@localhost/aragora",
        redis_url="redis://localhost:6379",
    )

    mound = KnowledgeMound(config, workspace_id="enterprise_team")
    await mound.initialize()

    # Store knowledge with provenance
    result = await mound.store(IngestionRequest(
        content="All contracts must have 90-day notice periods",
        source_type=KnowledgeSource.DEBATE,
        debate_id="debate_123",
        confidence=0.95,
        workspace_id="enterprise_team",
    ))

    # Query semantically
    results = await mound.query("contract notice requirements", limit=10)

    # Check staleness
    stale = await mound.get_stale_knowledge(threshold=0.7)

    # Get culture profile
    culture = await mound.get_culture_profile()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from aragora.config import DB_KNOWLEDGE_PATH
from aragora.knowledge.mound.types import (
    ConfidenceLevel,
    CulturePattern,
    CulturePatternType,
    CultureProfile,
    GraphQueryResult,
    IngestionRequest,
    IngestionResult,
    KnowledgeItem,
    KnowledgeLink,
    KnowledgeSource,
    MoundBackend,
    MoundConfig,
    MoundStats,
    QueryFilters,
    QueryResult,
    RelationshipType,
    SourceFilter,
    StalenessCheck,
    StalenessReason,
    StoreResult,
    SyncResult,
)
from aragora.knowledge.mound.converters import (
    node_to_item,
    relationship_to_link,
    continuum_to_item,
    consensus_to_item,
    fact_to_item,
    vector_result_to_item,
    evidence_to_item,
    critique_to_item,
)

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory
    from aragora.memory.consensus import ConsensusMemory
    from aragora.knowledge.fact_store import FactStore
    from aragora.evidence.store import EvidenceStore
    from aragora.memory.store import CritiqueStore
    from aragora.knowledge.mound.culture import (
        CultureDocument,
        OrganizationCulture,
        OrganizationCultureManager,
    )
    from aragora.rlm.types import RLMContext

# Check for RLM availability
try:
    from aragora.rlm import HierarchicalCompressor, RLMConfig, AbstractionLevel, RLMContext
    HAS_RLM = True
except ImportError:
    HAS_RLM = False
    HierarchicalCompressor = None
    RLMConfig = None
    AbstractionLevel = None

logger = logging.getLogger(__name__)


class KnowledgeMound:
    """
    Unified knowledge facade for the Aragora multi-agent control plane.

    The Knowledge Mound implements the "termite mound" architecture where
    all agents contribute to and query from a shared knowledge superstructure.

    Features:
    - Unified API across SQLite (dev), PostgreSQL (prod), and Redis (cache)
    - Cross-system queries across ContinuumMemory, ConsensusMemory, FactStore
    - Provenance tracking for audit and compliance
    - Staleness detection with automatic revalidation scheduling
    - Culture accumulation for organizational learning
    - Multi-tenant workspace isolation
    """

    def __init__(
        self,
        config: Optional[MoundConfig] = None,
        workspace_id: Optional[str] = None,
    ):
        """
        Initialize the Knowledge Mound.

        Args:
            config: Mound configuration. Defaults to SQLite backend.
            workspace_id: Default workspace for queries. Overrides config.
        """
        self.config = config or MoundConfig()
        self.workspace_id = workspace_id or self.config.default_workspace_id

        # Storage backends (initialized lazily)
        self._meta_store: Optional[Any] = None  # SQLite or Postgres
        self._cache: Optional[Any] = None  # Redis cache
        self._vector_store: Optional[Any] = None  # Weaviate
        self._semantic_store: Optional[Any] = None  # Local semantic index

        # Connected memory systems
        self._continuum: Optional["ContinuumMemory"] = None
        self._consensus: Optional["ConsensusMemory"] = None
        self._facts: Optional["FactStore"] = None
        self._evidence: Optional["EvidenceStore"] = None
        self._critique: Optional["CritiqueStore"] = None

        # Staleness detector and culture accumulator
        self._staleness_detector: Optional[Any] = None
        self._culture_accumulator: Optional[Any] = None

        # State
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize storage backends and connections."""
        if self._initialized:
            return

        logger.info(f"Initializing Knowledge Mound (backend={self.config.backend.value})")

        # Initialize primary storage based on backend
        if self.config.backend == MoundBackend.SQLITE:
            await self._init_sqlite()
        elif self.config.backend == MoundBackend.POSTGRES:
            await self._init_postgres()
        elif self.config.backend == MoundBackend.HYBRID:
            await self._init_postgres()
            await self._init_redis()

        # Initialize Redis cache if configured
        if self.config.redis_url and self.config.backend != MoundBackend.HYBRID:
            await self._init_redis()

        # Initialize vector store if configured
        if self.config.weaviate_url:
            await self._init_weaviate()

        # Initialize staleness detector
        if self.config.enable_staleness_detection:
            from aragora.knowledge.mound.staleness import StalenessDetector

            self._staleness_detector = StalenessDetector(
                mound=self,
                age_threshold=self.config.staleness_age_threshold,
            )

        # Initialize culture accumulator
        if self.config.enable_culture_accumulator:
            from aragora.knowledge.mound.culture import CultureAccumulator

            self._culture_accumulator = CultureAccumulator(mound=self)

        # Initialize semantic store for local embeddings
        await self._init_semantic_store()

        self._initialized = True
        logger.info("Knowledge Mound initialized successfully")

    async def _init_sqlite(self) -> None:
        """Initialize SQLite backend."""
        from aragora.knowledge.mound import KnowledgeMoundMetaStore

        db_path = self.config.sqlite_path or str(DB_KNOWLEDGE_PATH / "mound.db")
        self._meta_store = KnowledgeMoundMetaStore(db_path)
        logger.debug(f"SQLite backend initialized at {db_path}")

    async def _init_postgres(self) -> None:
        """Initialize PostgreSQL backend."""
        if not self.config.postgres_url:
            logger.warning("PostgreSQL URL not configured, falling back to SQLite")
            await self._init_sqlite()
            return

        try:
            from aragora.knowledge.mound.postgres_store import PostgresStore

            self._meta_store = PostgresStore(
                url=self.config.postgres_url,
                pool_size=self.config.postgres_pool_size,
                max_overflow=self.config.postgres_pool_max_overflow,
            )
            await self._meta_store.initialize()
            logger.debug("PostgreSQL backend initialized")
        except ImportError:
            logger.warning("asyncpg not available, falling back to SQLite")
            await self._init_sqlite()
        except Exception as e:
            logger.warning(f"PostgreSQL init failed: {e}, falling back to SQLite")
            await self._init_sqlite()

    async def _init_redis(self) -> None:
        """Initialize Redis cache."""
        if not self.config.redis_url:
            return

        try:
            from aragora.knowledge.mound.redis_cache import RedisCache

            self._cache = RedisCache(
                url=self.config.redis_url,
                default_ttl=self.config.redis_cache_ttl,
                culture_ttl=self.config.redis_culture_ttl,
            )
            await self._cache.connect()
            logger.debug("Redis cache initialized")
        except ImportError:
            logger.warning("redis not available, caching disabled")
        except Exception as e:
            logger.warning(f"Redis init failed: {e}, caching disabled")

    async def _init_weaviate(self) -> None:
        """Initialize Weaviate vector store."""
        try:
            from aragora.documents.indexing.weaviate_store import WeaviateStore, WeaviateConfig

            config = WeaviateConfig(
                url=self.config.weaviate_url,
                api_key=self.config.weaviate_api_key,
                collection_name=self.config.weaviate_collection,
            )
            self._vector_store = WeaviateStore(config)
            await self._vector_store.connect()
            logger.debug("Weaviate vector store initialized")
        except ImportError:
            logger.warning("Weaviate not available")
        except Exception as e:
            logger.warning(f"Weaviate init failed: {e}")

    async def _init_semantic_store(self) -> None:
        """Initialize local semantic store for embeddings."""
        try:
            from aragora.knowledge.mound.semantic_store import SemanticStore

            db_path = self.config.sqlite_path or str(DB_KNOWLEDGE_PATH / "mound.db")
            # Use a separate database for semantic index
            semantic_db_path = db_path.replace(".db", "_semantic.db")
            self._semantic_store = SemanticStore(
                db_path=semantic_db_path,
                default_tenant_id=self.workspace_id,
            )
            self._semantic_store.initialize()
            logger.debug(f"Semantic store initialized at {semantic_db_path}")
        except ImportError:
            logger.warning("Semantic store dependencies not available")
        except Exception as e:
            logger.warning(f"Semantic store init failed: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure the mound is initialized."""
        if not self._initialized:
            raise RuntimeError("KnowledgeMound not initialized. Call initialize() first.")

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def store(self, request: IngestionRequest) -> IngestionResult:
        """
        Store a new knowledge item with full provenance tracking.

        Args:
            request: Ingestion request with content and metadata

        Returns:
            IngestionResult with node ID and status
        """
        self._ensure_initialized()

        # Generate node ID
        node_id = f"kn_{uuid.uuid4().hex[:16]}"

        # Check for duplicates if enabled
        if self.config.enable_deduplication:
            content_hash = hashlib.sha256(request.content.encode()).hexdigest()[:32]
            existing = await self._find_by_content_hash(content_hash, request.workspace_id)
            if existing:
                # Update existing node
                await self._increment_update_count(existing)
                return IngestionResult(
                    node_id=existing,
                    success=True,
                    deduplicated=True,
                    existing_node_id=existing,
                    message="Merged with existing node",
                )

        # Create node data
        node_data = {
            "id": node_id,
            "workspace_id": request.workspace_id,
            "node_type": request.node_type,
            "content": request.content,
            "content_hash": hashlib.sha256(request.content.encode()).hexdigest()[:32],
            "confidence": request.confidence,
            "tier": request.tier,
            "source_type": request.source_type.value,
            "document_id": request.document_id,
            "debate_id": request.debate_id,
            "agent_id": request.agent_id,
            "user_id": request.user_id,
            "topics": request.topics,
            "metadata": request.metadata,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

        # Save to store
        await self._save_node(node_data)

        # Index in semantic store for embedding-based search
        if self._semantic_store:
            try:
                await self._semantic_store.index_item(
                    source_type=request.source_type,
                    source_id=node_id,
                    content=request.content,
                    tenant_id=request.workspace_id,
                    domain=request.topics[0] if request.topics else "general",
                    importance=request.confidence,
                )
            except Exception as e:
                logger.warning(f"Failed to index in semantic store: {e}")

        # Create relationships
        relationships_created = 0
        for target_id in request.supports:
            await self._save_relationship(node_id, target_id, "supports")
            relationships_created += 1
        for target_id in request.contradicts:
            await self._save_relationship(node_id, target_id, "contradicts")
            relationships_created += 1
        for target_id in request.derived_from:
            await self._save_relationship(node_id, target_id, "derived_from")
            relationships_created += 1

        # Invalidate cache
        if self._cache:
            await self._cache.invalidate_queries(request.workspace_id)

        logger.debug(f"Stored knowledge node: {node_id}")

        return IngestionResult(
            node_id=node_id,
            success=True,
            relationships_created=relationships_created,
        )

    async def get(self, node_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge node by ID."""
        self._ensure_initialized()

        # Check cache first
        if self._cache:
            cached = await self._cache.get_node(node_id)
            if cached:
                return cached

        # Query store
        node = await self._get_node(node_id)

        # Cache result
        if self._cache and node:
            await self._cache.set_node(node_id, node)

        return node

    async def update(self, node_id: str, updates: Dict[str, Any]) -> Optional[KnowledgeItem]:
        """Update a knowledge node."""
        self._ensure_initialized()

        updates["updated_at"] = datetime.now().isoformat()
        await self._update_node(node_id, updates)

        # Invalidate cache
        if self._cache:
            await self._cache.invalidate_node(node_id)

        return await self.get(node_id)

    async def delete(self, node_id: str, archive: bool = True) -> bool:
        """Delete a knowledge node."""
        self._ensure_initialized()

        if archive:
            await self._archive_node(node_id)

        result = await self._delete_node(node_id)

        # Invalidate cache
        if self._cache:
            await self._cache.invalidate_node(node_id)

        return result

    async def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        node_type: str = "fact",
        confidence: float = 0.7,
        tier: str = "medium",
    ) -> str:
        """
        Simplified method to add content to the Knowledge Mound.

        This is a convenience wrapper around store() for simpler use cases
        like repository crawling, document indexing, etc.

        Args:
            content: The content to store
            metadata: Optional metadata dictionary
            workspace_id: Workspace ID (defaults to self.workspace_id)
            node_type: Type of knowledge node (default: "fact")
            confidence: Confidence score 0-1 (default: 0.7)
            tier: Memory tier (default: "medium")

        Returns:
            The created node ID
        """
        request = IngestionRequest(
            content=content,
            workspace_id=workspace_id or self.workspace_id,
            node_type=node_type,
            confidence=confidence,
            tier=tier,
            source_type=KnowledgeSource.EXTERNAL,
            metadata=metadata or {},
        )
        result = await self.store(request)
        return result.node_id

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def query(
        self,
        query: str,
        sources: Sequence[SourceFilter] = ("all",),
        filters: Optional[QueryFilters] = None,
        limit: int = 20,
        workspace_id: Optional[str] = None,
    ) -> QueryResult:
        """
        Query across all configured knowledge sources.

        Args:
            query: Natural language query string
            sources: Which sources to query ("all" or specific sources)
            filters: Optional filters to apply
            limit: Maximum number of results
            workspace_id: Workspace to query (defaults to self.workspace_id)

        Returns:
            QueryResult with items from all queried sources
        """
        self._ensure_initialized()

        start_time = time.time()
        ws_id = workspace_id or self.workspace_id
        limit = min(limit, self.config.max_query_limit)

        # Check cache first
        cache_key = f"{ws_id}:{query}:{limit}:{sources}"
        if self._cache:
            cached = await self._cache.get_query(cache_key)
            if cached:
                return cached

        # Query local mound
        items = await self._query_local(query, filters, limit, ws_id)

        # Query connected memory systems in parallel
        if self.config.parallel_queries:
            tasks = []
            if "all" in sources or "continuum" in sources:
                tasks.append(self._query_continuum(query, filters, limit))
            if "all" in sources or "consensus" in sources:
                tasks.append(self._query_consensus(query, filters, limit))
            if "all" in sources or "fact" in sources:
                tasks.append(self._query_facts(query, filters, limit, ws_id))
            if "all" in sources or "evidence" in sources:
                tasks.append(self._query_evidence(query, filters, limit, ws_id))
            if "all" in sources or "critique" in sources:
                tasks.append(self._query_critique(query, filters, limit))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        items.extend(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Query source failed: {result}")

        # Sort by importance/relevance and limit
        items.sort(key=lambda x: x.importance or 0, reverse=True)
        items = items[:limit]

        execution_time = (time.time() - start_time) * 1000

        result = QueryResult(
            items=items,
            total_count=len(items),
            query=query,
            filters=filters,
            execution_time_ms=execution_time,
            sources_queried=[KnowledgeSource(s) for s in sources if s != "all"],
        )

        # Cache result
        if self._cache:
            await self._cache.set_query(cache_key, result)

        return result

    async def query_semantic(
        self,
        text: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        workspace_id: Optional[str] = None,
    ) -> List[KnowledgeItem]:
        """Semantic similarity search using vector embeddings."""
        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id

        # Try Weaviate first (production vector store)
        if self._vector_store:
            try:
                results = await self._vector_store.search(
                    query=text,
                    limit=limit,
                    filters={"workspace_id": ws_id},
                )
                return [self._vector_result_to_item(r) for r in results]
            except Exception as e:
                logger.warning(f"Weaviate search failed: {e}, falling back")

        # Try local semantic store (embeddings in SQLite)
        if self._semantic_store:
            try:
                results = await self._semantic_store.search_similar(
                    query=text,
                    tenant_id=ws_id,
                    limit=limit,
                    min_similarity=min_confidence,
                )
                # Convert semantic results to KnowledgeItems
                items = []
                for sr in results:
                    node = await self.get(sr.source_id)
                    if node:
                        items.append(node)
                return items
            except Exception as e:
                logger.warning(f"Semantic store search failed: {e}, falling back")

        # Fall back to keyword search
        result = await self.query(text, limit=limit, workspace_id=workspace_id)
        return result.items

    async def query_graph(
        self,
        start_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        depth: int = 2,
        max_nodes: int = 50,
    ) -> GraphQueryResult:
        """Traverse knowledge graph from a starting node."""
        self._ensure_initialized()

        nodes: Dict[str, KnowledgeItem] = {}
        edges: List[KnowledgeLink] = []
        visited: set = set()

        async def traverse(node_id: str, current_depth: int) -> None:
            if current_depth > depth or node_id in visited or len(nodes) >= max_nodes:
                return

            visited.add(node_id)
            node = await self.get(node_id)
            if node:
                nodes[node_id] = node

                if current_depth < depth:
                    relationships = await self._get_relationships(node_id, relationship_types)
                    for rel in relationships:
                        edges.append(rel)
                        target = rel.target_id if rel.source_id == node_id else rel.source_id
                        if target not in visited:
                            await traverse(target, current_depth + 1)

        await traverse(start_id, 0)

        return GraphQueryResult(
            nodes=list(nodes.values()),
            edges=edges,
            root_id=start_id,
            depth=depth,
            total_nodes=len(nodes),
            total_edges=len(edges),
        )

    # =========================================================================
    # RLM Integration (Recursive Language Models)
    # =========================================================================

    async def query_with_rlm(
        self,
        query: str,
        limit: int = 50,
        workspace_id: Optional[str] = None,
        agent_call: Optional[Any] = None,
    ) -> Optional["RLMContext"]:
        """
        Query knowledge and build hierarchical RLM context for navigation.

        Based on the "Recursive Language Models" paper (arXiv:2512.24601),
        this method builds a hierarchical representation of query results
        that enables efficient navigation from summaries to details.

        Args:
            query: Semantic query text
            limit: Maximum knowledge items to include
            workspace_id: Workspace to query
            agent_call: Optional callback for LLM-based compression

        Returns:
            RLMContext with hierarchical representation of knowledge,
            or None if RLM is not available.

        Example:
            ctx = await mound.query_with_rlm("contract requirements", limit=30)
            if ctx:
                # Get high-level overview
                abstract = ctx.get_at_level(AbstractionLevel.ABSTRACT)

                # Drill into specific node
                details = ctx.drill_down("type_fact")
        """
        if not HAS_RLM:
            logger.warning("RLM not available, use query_semantic instead")
            return None

        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id

        # Fetch relevant knowledge items
        items = await self.query_semantic(
            text=query,
            limit=limit,
            workspace_id=ws_id,
        )

        if not items:
            logger.debug("No knowledge items found for RLM context")
            return None

        # Build text content from knowledge items
        content_parts = []
        for item in items:
            item_text = f"[{item.id}] ({item.source_type.value if item.source_type else 'unknown'})\n"
            item_text += f"**Confidence**: {item.confidence:.0%}\n"
            item_text += f"{item.content}\n"
            content_parts.append(item_text)

        full_content = "\n---\n".join(content_parts)

        # Create compressor
        config = RLMConfig() if RLMConfig else None
        compressor = HierarchicalCompressor(
            config=config,
            agent_call=agent_call,
        ) if HierarchicalCompressor else None

        if not compressor:
            logger.warning("Failed to create RLM compressor")
            return None

        # Compress into hierarchical context
        try:
            result = await compressor.compress(
                content=full_content,
                source_type="knowledge",
                max_levels=3,
            )

            logger.info(
                "[rlm] Built hierarchical context from %d knowledge items "
                "(%d tokens → %d levels)",
                len(items),
                result.original_tokens,
                len(result.context.levels),
            )

            return result.context

        except Exception as e:
            logger.error(f"RLM compression failed: {e}")
            return None

    def is_rlm_available(self) -> bool:
        """Check if RLM features are available."""
        return HAS_RLM

    # =========================================================================
    # Staleness Management
    # =========================================================================

    async def get_stale_knowledge(
        self,
        threshold: float = 0.5,
        limit: int = 100,
        workspace_id: Optional[str] = None,
    ) -> List[StalenessCheck]:
        """Get knowledge items that may be stale."""
        self._ensure_initialized()

        if not self._staleness_detector:
            return []

        ws_id = workspace_id or self.workspace_id
        return await self._staleness_detector.get_stale_nodes(
            workspace_id=ws_id,
            threshold=threshold,
            limit=limit,
        )

    async def mark_validated(
        self,
        node_id: str,
        validator: str,
        confidence: Optional[float] = None,
    ) -> None:
        """Mark a knowledge node as validated."""
        self._ensure_initialized()

        updates = {
            "validation_status": "verified",
            "last_validated_at": datetime.now().isoformat(),
            "staleness_score": 0.0,
        }
        if confidence is not None:
            updates["confidence"] = confidence

        await self.update(node_id, updates)

    async def schedule_revalidation(
        self,
        node_ids: List[str],
        priority: str = "low",
    ) -> List[str]:
        """
        Schedule nodes for revalidation via the control plane.

        Creates tasks in the control plane task queue for each node
        that needs revalidation. Workers will pick up these tasks
        and run revalidation debates/checks.

        Args:
            node_ids: List of node IDs to revalidate
            priority: Task priority ("low", "normal", "high")

        Returns:
            List of created task IDs
        """
        self._ensure_initialized()

        task_ids = []
        now = datetime.now().isoformat()

        for node_id in node_ids:
            # Mark node as needing revalidation
            await self.update(node_id, {"revalidation_requested": True})

            # Create control plane task
            task_id = f"reval_{uuid.uuid4().hex[:12]}"
            task = {
                "id": task_id,
                "type": "knowledge_revalidation",
                "priority": priority,
                "status": "pending",
                "node_id": node_id,
                "workspace_id": self.workspace_id,
                "created_at": now,
                "metadata": {
                    "source": "knowledge_mound",
                    "action": "revalidate",
                },
            }

            # Add to control plane queue
            try:
                from aragora.server.handlers.features.control_plane import _task_queue
                _task_queue.append(task)
                task_ids.append(task_id)
                logger.debug(f"Scheduled revalidation task {task_id} for node {node_id}")
            except ImportError:
                # Control plane not available, just log
                logger.warning(
                    f"Control plane not available, revalidation for {node_id} marked but not queued"
                )
                task_ids.append(f"pending_{node_id}")

        logger.info(f"Scheduled {len(task_ids)} revalidation tasks with priority={priority}")
        return task_ids

    # =========================================================================
    # Culture Accumulation
    # =========================================================================

    async def get_culture_profile(
        self,
        workspace_id: Optional[str] = None,
    ) -> CultureProfile:
        """Get aggregated culture profile for a workspace."""
        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id

        # Check cache
        if self._cache:
            cached = await self._cache.get_culture(ws_id)
            if cached:
                return cached

        if not self._culture_accumulator:
            return CultureProfile(
                workspace_id=ws_id,
                patterns={},
                generated_at=datetime.now(),
                total_observations=0,
            )

        profile = await self._culture_accumulator.get_profile(ws_id)

        # Cache result
        if self._cache:
            await self._cache.set_culture(ws_id, profile)

        return profile

    async def observe_debate(self, debate_result: Any) -> List[CulturePattern]:
        """Extract and store cultural patterns from a completed debate."""
        self._ensure_initialized()

        if not self._culture_accumulator:
            return []

        return await self._culture_accumulator.observe_debate(
            debate_result, self.workspace_id
        )

    async def recommend_agents(
        self,
        task_type: str,
        workspace_id: Optional[str] = None,
    ) -> List[str]:
        """Recommend agents based on cultural patterns."""
        self._ensure_initialized()

        if not self._culture_accumulator:
            return []

        ws_id = workspace_id or self.workspace_id
        return await self._culture_accumulator.recommend_agents(task_type, ws_id)

    # =========================================================================
    # Organization-Level Culture
    # =========================================================================

    def get_org_culture_manager(self) -> "OrganizationCultureManager":
        """
        Get the organization culture manager.

        Returns:
            OrganizationCultureManager instance
        """
        self._ensure_initialized()

        if not hasattr(self, "_org_culture_manager") or self._org_culture_manager is None:
            from aragora.knowledge.mound.culture import OrganizationCultureManager

            self._org_culture_manager = OrganizationCultureManager(
                mound=self,
                culture_accumulator=self._culture_accumulator,
            )

        return self._org_culture_manager

    async def get_org_culture(
        self,
        org_id: str,
        workspace_ids: Optional[List[str]] = None,
    ) -> "OrganizationCulture":
        """
        Get the organization culture profile.

        Aggregates patterns from all workspaces plus explicit culture documents.

        Args:
            org_id: Organization ID
            workspace_ids: Optional list of workspaces to include

        Returns:
            Complete organization culture profile
        """
        manager = self.get_org_culture_manager()
        return await manager.get_organization_culture(org_id, workspace_ids)

    async def add_culture_document(
        self,
        org_id: str,
        category: str,
        title: str,
        content: str,
        created_by: str,
    ) -> "CultureDocument":
        """
        Add an explicit culture document.

        Args:
            org_id: Organization ID
            category: Document category (values, practices, standards, policies, learnings)
            title: Document title
            content: Document content
            created_by: User creating the document

        Returns:
            Created culture document
        """
        from aragora.knowledge.mound.culture import CultureDocumentCategory

        manager = self.get_org_culture_manager()
        return await manager.add_document(
            org_id=org_id,
            category=CultureDocumentCategory(category),
            title=title,
            content=content,
            created_by=created_by,
        )

    async def promote_to_culture(
        self,
        workspace_id: str,
        pattern_id: str,
        promoted_by: str,
        title: Optional[str] = None,
    ) -> "CultureDocument":
        """
        Promote a workspace pattern to organization culture.

        Args:
            workspace_id: Workspace containing the pattern
            pattern_id: Pattern ID to promote
            promoted_by: User promoting the pattern
            title: Optional title override

        Returns:
            Created culture document
        """
        manager = self.get_org_culture_manager()
        return await manager.promote_pattern_to_culture(
            workspace_id=workspace_id,
            pattern_id=pattern_id,
            promoted_by=promoted_by,
            title=title,
        )

    async def get_culture_context(
        self,
        org_id: str,
        task: str,
        max_documents: int = 3,
    ) -> str:
        """
        Get relevant culture context for a task.

        This is used to inject organizational knowledge into agent prompts.

        Args:
            org_id: Organization ID
            task: Task description
            max_documents: Maximum documents to include

        Returns:
            Formatted context string
        """
        manager = self.get_org_culture_manager()
        return await manager.get_relevant_context(org_id, task, max_documents)

    def register_workspace_org(self, workspace_id: str, org_id: str) -> None:
        """
        Register a workspace's organization for culture aggregation.

        Args:
            workspace_id: Workspace ID
            org_id: Organization ID
        """
        manager = self.get_org_culture_manager()
        manager.register_workspace(workspace_id, org_id)

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def sync_from_continuum(
        self,
        continuum: "ContinuumMemory",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> SyncResult:
        """
        Sync knowledge from ContinuumMemory.

        Iterates through memory entries and stores them as knowledge nodes.
        Uses content hash deduplication to avoid duplicates.

        Args:
            continuum: ContinuumMemory instance to sync from
            incremental: If True, only sync entries updated since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        self._ensure_initialized()

        start_time = time.time()
        self._continuum = continuum
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            # Retrieve all entries from continuum (using high limit for full sync)
            entries = continuum.retrieve(
                query=None,
                tiers=None,
                limit=10000,  # High limit for sync
                min_importance=0.0,
                include_glacial=True,
            )

            for entry in entries:
                try:
                    # Create ingestion request from continuum entry
                    request = IngestionRequest(
                        content=entry.content,
                        workspace_id=self.workspace_id,
                        source_type=KnowledgeSource.CONTINUUM,
                        node_type="memory",
                        confidence=entry.importance,
                        tier=entry.tier.value,
                        metadata={
                            "continuum_id": entry.id,
                            "surprise_score": entry.surprise_score,
                            "consolidation_score": entry.consolidation_score,
                            "update_count": entry.update_count,
                            "success_rate": entry.success_rate,
                            "original_metadata": entry.metadata,
                        },
                    )

                    result = await self.store(request)

                    if result.deduplicated:
                        nodes_updated += 1
                    else:
                        nodes_synced += 1

                except Exception as e:
                    nodes_skipped += 1
                    errors.append(f"continuum:{entry.id}: {str(e)}")
                    logger.warning(f"Failed to sync continuum entry {entry.id}: {e}")

        except Exception as e:
            errors.append(f"continuum:retrieve: {str(e)}")
            logger.error(f"Failed to retrieve continuum entries: {e}")

        return SyncResult(
            source="continuum",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_consensus(
        self,
        consensus: "ConsensusMemory",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> SyncResult:
        """
        Sync knowledge from ConsensusMemory.

        Stores consensus records as high-confidence knowledge nodes.

        Args:
            consensus: ConsensusMemory instance to sync from
            incremental: If True, only sync entries since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        self._ensure_initialized()

        start_time = time.time()
        self._consensus = consensus
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        relationships_created = 0
        errors: List[str] = []

        try:
            # Get recent consensus records from the store
            # ConsensusMemory stores records in SQLite, we query directly
            if hasattr(consensus, '_store') and consensus._store:
                with consensus._store.connection() as conn:
                    cursor = conn.execute(
                        """
                        SELECT id, topic, conclusion, strength, confidence,
                               participating_agents, agreeing_agents, domain, tags,
                               timestamp, supersedes, metadata
                        FROM consensus_records
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (10000,),
                    )
                    rows = cursor.fetchall()

                for row in rows:
                    try:
                        record_id = row[0]
                        topic = row[1]
                        conclusion = row[2]
                        strength = row[3]
                        confidence = row[4]
                        domain = row[7]
                        tags_json = row[8]
                        supersedes = row[10]
                        metadata_json = row[11]

                        # Parse JSON fields
                        from aragora.utils.json_helpers import safe_json_loads
                        tags = safe_json_loads(tags_json, [])
                        metadata = safe_json_loads(metadata_json, {})

                        # Create ingestion request
                        request = IngestionRequest(
                            content=f"{topic}: {conclusion}",
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.CONSENSUS,
                            debate_id=record_id,
                            node_type="consensus",
                            confidence=confidence,
                            tier="slow",  # Consensus is stable knowledge
                            topics=tags,
                            metadata={
                                "consensus_id": record_id,
                                "strength": strength,
                                "domain": domain,
                                "original_metadata": metadata,
                            },
                        )

                        # Add supersession relationship
                        if supersedes:
                            request.derived_from = [f"cs_{supersedes}"]

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                        relationships_created += result.relationships_created

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"consensus:{row[0]}: {str(e)}")
                        logger.warning(f"Failed to sync consensus record {row[0]}: {e}")

        except Exception as e:
            errors.append(f"consensus:query: {str(e)}")
            logger.error(f"Failed to query consensus records: {e}")

        return SyncResult(
            source="consensus",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=relationships_created,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_facts(
        self,
        facts: "FactStore",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> SyncResult:
        """
        Sync knowledge from FactStore.

        Stores facts as knowledge nodes with evidence relationships.

        Args:
            facts: FactStore instance to sync from
            incremental: If True, only sync since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        self._ensure_initialized()

        start_time = time.time()
        self._facts = facts
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        relationships_created = 0
        errors: List[str] = []

        try:
            # FactStore has query_facts method
            if hasattr(facts, 'query_facts'):
                all_facts = facts.query_facts(
                    query="",
                    workspace_id=self.workspace_id,
                    limit=10000,
                )

                for fact in all_facts:
                    try:
                        request = IngestionRequest(
                            content=fact.statement,
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.FACT,
                            document_id=fact.source_documents[0] if fact.source_documents else None,
                            node_type="fact",
                            confidence=fact.confidence,
                            tier="slow",
                            topics=fact.topics,
                            metadata={
                                "fact_id": fact.id,
                                "validation_status": fact.validation_status.value if hasattr(fact.validation_status, 'value') else str(fact.validation_status),
                                "evidence_ids": fact.evidence_ids,
                                "source_documents": fact.source_documents,
                            },
                        )

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"facts:{fact.id}: {str(e)}")
                        logger.warning(f"Failed to sync fact {fact.id}: {e}")

        except Exception as e:
            errors.append(f"facts:query: {str(e)}")
            logger.error(f"Failed to query facts: {e}")

        return SyncResult(
            source="facts",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=relationships_created,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_evidence(
        self,
        evidence: "EvidenceStore",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> SyncResult:
        """
        Sync knowledge from EvidenceStore.

        Stores evidence snippets as knowledge nodes.

        Args:
            evidence: EvidenceStore instance to sync from
            incremental: If True, only sync since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        self._ensure_initialized()

        start_time = time.time()
        self._evidence = evidence
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            # EvidenceStore has search method
            if hasattr(evidence, 'search'):
                all_evidence = evidence.search("", limit=10000)

                for ev in all_evidence:
                    try:
                        request = IngestionRequest(
                            content=ev.content,
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.EVIDENCE,
                            debate_id=getattr(ev, 'debate_id', None),
                            agent_id=getattr(ev, 'agent_id', None),
                            node_type="evidence",
                            confidence=getattr(ev, 'quality_score', 0.5),
                            tier="medium",
                            metadata={
                                "evidence_id": ev.id,
                                "source_url": getattr(ev, 'source_url', None),
                            },
                        )

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"evidence:{ev.id}: {str(e)}")
                        logger.warning(f"Failed to sync evidence {ev.id}: {e}")

        except Exception as e:
            errors.append(f"evidence:search: {str(e)}")
            logger.error(f"Failed to search evidence: {e}")

        return SyncResult(
            source="evidence",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_from_critique(
        self,
        critique: "CritiqueStore",
        incremental: bool = True,
        batch_size: int = 100,
    ) -> SyncResult:
        """
        Sync knowledge from CritiqueStore (critique patterns).

        Stores successful critique patterns as knowledge nodes.

        Args:
            critique: CritiqueStore instance to sync from
            incremental: If True, only sync since last sync
            batch_size: Number of entries to process per batch

        Returns:
            SyncResult with counts of synced/updated/skipped nodes
        """
        self._ensure_initialized()

        start_time = time.time()
        self._critique = critique
        nodes_synced = 0
        nodes_updated = 0
        nodes_skipped = 0
        errors: List[str] = []

        try:
            # CritiqueStore has search_patterns method
            if hasattr(critique, 'search_patterns'):
                patterns = critique.search_patterns("", limit=10000)

                for pattern in patterns:
                    try:
                        content = getattr(pattern, 'pattern', '') or getattr(pattern, 'content', '')
                        if not content:
                            nodes_skipped += 1
                            continue

                        request = IngestionRequest(
                            content=content,
                            workspace_id=self.workspace_id,
                            source_type=KnowledgeSource.CRITIQUE,
                            agent_id=getattr(pattern, 'agent_name', None),
                            node_type="critique",
                            confidence=getattr(pattern, 'success_rate', 0.5),
                            tier="slow",
                            metadata={
                                "pattern_id": pattern.id,
                                "success_count": getattr(pattern, 'success_count', 0),
                            },
                        )

                        result = await self.store(request)

                        if result.deduplicated:
                            nodes_updated += 1
                        else:
                            nodes_synced += 1

                    except Exception as e:
                        nodes_skipped += 1
                        errors.append(f"critique:{pattern.id}: {str(e)}")
                        logger.warning(f"Failed to sync critique pattern {pattern.id}: {e}")

        except Exception as e:
            errors.append(f"critique:search: {str(e)}")
            logger.error(f"Failed to search critique patterns: {e}")

        return SyncResult(
            source="critique",
            nodes_synced=nodes_synced,
            nodes_updated=nodes_updated,
            nodes_skipped=nodes_skipped,
            relationships_created=0,
            duration_ms=int((time.time() - start_time) * 1000),
            errors=errors,
        )

    async def sync_all(self) -> Dict[str, SyncResult]:
        """
        Sync from all connected memory systems.

        Returns a dict mapping source name to SyncResult.
        Only syncs from sources that have been connected.
        """
        self._ensure_initialized()
        results: Dict[str, SyncResult] = {}

        if self._continuum:
            results["continuum"] = await self.sync_from_continuum(self._continuum)

        if self._consensus:
            results["consensus"] = await self.sync_from_consensus(self._consensus)

        if self._facts:
            results["facts"] = await self.sync_from_facts(self._facts)

        if self._evidence:
            results["evidence"] = await self.sync_from_evidence(self._evidence)

        if self._critique:
            results["critique"] = await self.sync_from_critique(self._critique)

        logger.info(
            "Sync complete: %d sources, %d total nodes synced",
            len(results),
            sum(r.nodes_synced for r in results.values()),
        )

        return results

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self, workspace_id: Optional[str] = None) -> MoundStats:
        """Get statistics about the Knowledge Mound."""
        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id
        return await self._get_stats(ws_id)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close all connections."""
        if self._cache:
            await self._cache.close()
        if self._vector_store:
            try:
                await self._vector_store.close()
            except Exception:
                pass
        if hasattr(self._meta_store, "close"):
            await self._meta_store.close()

        self._initialized = False
        logger.info("Knowledge Mound closed")

    @asynccontextmanager
    async def session(self) -> AsyncIterator["KnowledgeMound"]:
        """Context manager for managed lifecycle."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()

    # =========================================================================
    # Private Implementation Methods
    # =========================================================================

    async def _save_node(self, node_data: Dict[str, Any]) -> None:
        """Save node to storage."""
        if hasattr(self._meta_store, "save_node_async"):
            await self._meta_store.save_node_async(node_data)
        else:
            # SQLite sync fallback
            from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType

            # Map KnowledgeSource values to valid ProvenanceType values
            source_to_provenance = {
                "document": ProvenanceType.DOCUMENT,
                "debate": ProvenanceType.DEBATE,
                "consensus": ProvenanceType.DEBATE,  # Consensus comes from debates
                "user": ProvenanceType.USER,
                "fact": ProvenanceType.AGENT,  # Facts are agent-derived
                "continuum": ProvenanceType.INFERENCE,  # Memory-derived
                "vector": ProvenanceType.DOCUMENT,  # Vector embeddings from documents
                "external": ProvenanceType.MIGRATION,  # External sources
                "extraction": ProvenanceType.AGENT,  # Agent extraction
            }

            node = KnowledgeNode(
                id=node_data["id"],
                node_type=node_data["node_type"],
                content=node_data["content"],
                confidence=node_data["confidence"],
                workspace_id=node_data["workspace_id"],
                metadata=node_data.get("metadata", {}),
                topics=node_data.get("topics", []),
            )
            if node_data.get("source_type"):
                source_type_str = node_data["source_type"]
                provenance_type = source_to_provenance.get(
                    source_type_str, ProvenanceType.DOCUMENT
                )
                node.provenance = ProvenanceChain(
                    source_type=provenance_type,
                    source_id=node_data.get("debate_id") or node_data.get("document_id") or "",
                    debate_id=node_data.get("debate_id"),
                    document_id=node_data.get("document_id"),
                    agent_id=node_data.get("agent_id"),
                    user_id=node_data.get("user_id"),
                )
            self._meta_store.save_node(node)

    async def _get_node(self, node_id: str) -> Optional[KnowledgeItem]:
        """Get node from storage."""
        if hasattr(self._meta_store, "get_node_async"):
            return await self._meta_store.get_node_async(node_id)
        else:
            node = self._meta_store.get_node(node_id)
            if node:
                return self._node_to_item(node)
            return None

    async def _update_node(self, node_id: str, updates: Dict[str, Any]) -> None:
        """Update node in storage."""
        # For SQLite, get then save
        if hasattr(self._meta_store, "update_node_async"):
            await self._meta_store.update_node_async(node_id, updates)
        else:
            node = self._meta_store.get_node(node_id)
            if node:
                for key, value in updates.items():
                    if hasattr(node, key):
                        setattr(node, key, value)
                self._meta_store.save_node(node)

    async def _delete_node(self, node_id: str) -> bool:
        """Delete node from storage."""
        if hasattr(self._meta_store, "delete_node_async"):
            return await self._meta_store.delete_node_async(node_id)
        else:
            # SQLite doesn't have delete, use raw SQL
            with self._meta_store.connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM knowledge_nodes WHERE id = ?", (node_id,)
                )
                return cursor.rowcount > 0

    async def _archive_node(self, node_id: str) -> None:
        """
        Archive node before deletion.

        Saves the node to an archive table/collection for audit trail
        and potential recovery. The archive includes full node data
        plus deletion metadata.
        """
        node = await self.get(node_id)
        if not node:
            logger.debug(f"Node {node_id} not found, skipping archive")
            return

        archive_record = {
            "id": f"arch_{node_id}_{uuid.uuid4().hex[:8]}",
            "original_id": node_id,
            "content": node.content,
            "source": node.source.value if hasattr(node.source, 'value') else str(node.source),
            "source_id": node.source_id,
            "confidence": node.confidence.value if hasattr(node.confidence, 'value') else str(node.confidence),
            "importance": node.importance,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "updated_at": node.updated_at.isoformat() if node.updated_at else None,
            "archived_at": datetime.now().isoformat(),
            "workspace_id": self.workspace_id,
        }

        # Save to archive store
        if hasattr(self._meta_store, "archive_node_async"):
            await self._meta_store.archive_node_async(archive_record)
        elif hasattr(self._meta_store, "archive_node"):
            self._meta_store.archive_node(archive_record)
        else:
            # Fallback: store in SQLite archive table
            try:
                with self._meta_store.connection() as conn:
                    # Create archive table if it doesn't exist
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS knowledge_archive (
                            id TEXT PRIMARY KEY,
                            original_id TEXT NOT NULL,
                            content TEXT NOT NULL,
                            source TEXT,
                            source_id TEXT,
                            confidence TEXT,
                            importance REAL,
                            metadata TEXT,
                            created_at TEXT,
                            updated_at TEXT,
                            archived_at TEXT NOT NULL,
                            workspace_id TEXT
                        )
                    """)
                    conn.execute("""
                        INSERT INTO knowledge_archive
                        (id, original_id, content, source, source_id, confidence,
                         importance, metadata, created_at, updated_at, archived_at, workspace_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        archive_record["id"],
                        archive_record["original_id"],
                        archive_record["content"],
                        archive_record["source"],
                        archive_record["source_id"],
                        archive_record["confidence"],
                        archive_record["importance"],
                        json.dumps(archive_record["metadata"]) if archive_record["metadata"] else "{}",
                        archive_record["created_at"],
                        archive_record["updated_at"],
                        archive_record["archived_at"],
                        archive_record["workspace_id"],
                    ))
                    conn.commit()
                logger.debug(f"Archived node {node_id} to knowledge_archive table")
            except Exception as e:
                logger.warning(f"Failed to archive node {node_id}: {e}")
                # Don't block deletion on archive failure

    async def _save_relationship(
        self, from_id: str, to_id: str, rel_type: str
    ) -> None:
        """Save relationship to storage."""
        if hasattr(self._meta_store, "save_relationship_async"):
            await self._meta_store.save_relationship_async(from_id, to_id, rel_type)
        else:
            from aragora.knowledge.mound import KnowledgeRelationship

            rel = KnowledgeRelationship(
                from_node_id=from_id,
                to_node_id=to_id,
                relationship_type=rel_type,
            )
            self._meta_store.save_relationship(rel)

    async def _get_relationships(
        self, node_id: str, types: Optional[List[RelationshipType]] = None
    ) -> List[KnowledgeLink]:
        """Get relationships for a node."""
        if hasattr(self._meta_store, "get_relationships_async"):
            return await self._meta_store.get_relationships_async(node_id, types)
        else:
            rels = self._meta_store.get_relationships(node_id)
            return [self._rel_to_link(r) for r in rels]

    async def _find_by_content_hash(
        self, content_hash: str, workspace_id: str
    ) -> Optional[str]:
        """Find node by content hash."""
        if hasattr(self._meta_store, "find_by_content_hash_async"):
            return await self._meta_store.find_by_content_hash_async(
                content_hash, workspace_id
            )
        else:
            node = self._meta_store.find_by_content_hash(content_hash, workspace_id)
            return node.id if node else None

    async def _increment_update_count(self, node_id: str) -> None:
        """Increment update count for a node."""
        await self._update_node(node_id, {"update_count": "update_count + 1"})

    async def _query_local(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
        workspace_id: str,
    ) -> List[KnowledgeItem]:
        """Query local mound storage."""
        if hasattr(self._meta_store, "query_async"):
            return await self._meta_store.query_async(query, filters, limit, workspace_id)
        else:
            nodes = self._meta_store.query_nodes(
                workspace_id=workspace_id,
                limit=limit,
            )
            # Simple keyword matching
            query_words = set(query.lower().split())
            scored = []
            for node in nodes:
                content_words = set(node.content.lower().split())
                score = len(query_words & content_words) / max(len(query_words), 1)
                scored.append((score, node))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [self._node_to_item(n) for _, n in scored[:limit]]

    async def _query_continuum(
        self, query: str, filters: Optional[QueryFilters], limit: int
    ) -> List[KnowledgeItem]:
        """Query ContinuumMemory."""
        if not self._continuum:
            return []
        try:
            entries = self._continuum.search_by_keyword(query, limit=limit)
            return [self._continuum_to_item(e) for e in entries]
        except Exception as e:
            logger.warning(f"Continuum query failed: {e}")
            return []

    async def _query_consensus(
        self, query: str, filters: Optional[QueryFilters], limit: int
    ) -> List[KnowledgeItem]:
        """Query ConsensusMemory."""
        if not self._consensus:
            return []
        try:
            entries = await self._consensus.search_by_topic(query, limit=limit)
            return [self._consensus_to_item(e) for e in entries]
        except Exception as e:
            logger.warning(f"Consensus query failed: {e}")
            return []

    async def _query_facts(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
        workspace_id: str,
    ) -> List[KnowledgeItem]:
        """Query FactStore."""
        if not self._facts:
            return []
        try:
            facts = self._facts.query_facts(query, workspace_id=workspace_id, limit=limit)
            return [self._fact_to_item(f) for f in facts]
        except Exception as e:
            logger.warning(f"Facts query failed: {e}")
            return []

    async def _query_evidence(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
        workspace_id: str,
    ) -> List[KnowledgeItem]:
        """Query EvidenceStore."""
        if not self._evidence:
            return []
        try:
            # EvidenceStore.search returns evidence snippets
            evidence_list = self._evidence.search(query, limit=limit)
            return [self._evidence_to_item(e) for e in evidence_list]
        except Exception as e:
            logger.warning(f"Evidence query failed: {e}")
            return []

    async def _query_critique(
        self,
        query: str,
        filters: Optional[QueryFilters],
        limit: int,
    ) -> List[KnowledgeItem]:
        """Query CritiqueStore for successful patterns."""
        if not self._critique:
            return []
        try:
            # CritiqueStore.search_patterns returns critique patterns
            patterns = self._critique.search_patterns(query, limit=limit)
            return [self._critique_to_item(p) for p in patterns]
        except Exception as e:
            logger.warning(f"Critique query failed: {e}")
            return []

    async def _get_stats(self, workspace_id: str) -> MoundStats:
        """Get statistics from storage."""
        if hasattr(self._meta_store, "get_stats_async"):
            return await self._meta_store.get_stats_async(workspace_id)
        else:
            stats = self._meta_store.get_stats(workspace_id)
            return MoundStats(
                total_nodes=stats.get("total_nodes", 0),
                nodes_by_type=stats.get("by_type", {}),
                nodes_by_tier=stats.get("by_tier", {}),
                nodes_by_validation=stats.get("by_validation_status", {}),
                total_relationships=stats.get("total_relationships", 0),
                relationships_by_type={},
                average_confidence=stats.get("average_confidence", 0.0),
                stale_nodes_count=0,
                workspace_id=workspace_id,
            )

    # =========================================================================
    # Conversion Helpers (delegated to converters module)
    # =========================================================================

    def _node_to_item(self, node: Any) -> KnowledgeItem:
        """Convert KnowledgeNode to KnowledgeItem."""
        return node_to_item(node)

    def _rel_to_link(self, rel: Any) -> KnowledgeLink:
        """Convert KnowledgeRelationship to KnowledgeLink."""
        return relationship_to_link(rel)

    def _continuum_to_item(self, entry: Any) -> KnowledgeItem:
        """Convert ContinuumMemory entry to KnowledgeItem."""
        return continuum_to_item(entry)

    def _consensus_to_item(self, entry: Any) -> KnowledgeItem:
        """Convert ConsensusMemory entry to KnowledgeItem."""
        return consensus_to_item(entry)

    def _fact_to_item(self, fact: Any) -> KnowledgeItem:
        """Convert Fact to KnowledgeItem."""
        return fact_to_item(fact)

    def _vector_result_to_item(self, result: Any) -> KnowledgeItem:
        """Convert vector search result to KnowledgeItem."""
        return vector_result_to_item(result)

    def _evidence_to_item(self, evidence: Any) -> KnowledgeItem:
        """Convert EvidenceSnippet to KnowledgeItem."""
        return evidence_to_item(evidence)

    def _critique_to_item(self, pattern: Any) -> KnowledgeItem:
        """Convert CritiquePattern to KnowledgeItem."""
        return critique_to_item(pattern)
