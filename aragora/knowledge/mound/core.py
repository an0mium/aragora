"""
Knowledge Mound Core - Base class with initialization and storage adapters.

Provides the foundation for KnowledgeMound:
- Constructor and initialization
- Storage backend initialization (SQLite, PostgreSQL, Redis, Weaviate)
- Private storage adapter methods
- Query helper methods for connected stores
- Lifecycle management (close, session)
- Statistics methods
- Converter wrappers
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from aragora.config import DB_KNOWLEDGE_PATH
from aragora.knowledge.mound.types import (
    KnowledgeItem,
    KnowledgeLink,
    MoundBackend,
    MoundConfig,
    MoundStats,
    QueryFilters,
    RelationshipType,
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
    from aragora.knowledge.mound.culture import OrganizationCultureManager
    from aragora.types.protocols import EventEmitterProtocol

logger = logging.getLogger(__name__)


class KnowledgeMoundCore:
    """
    Core foundation for the Knowledge Mound facade.

    Provides initialization, storage adapters, and utility methods
    that are used by the operation mixins.
    """

    def __init__(
        self,
        config: Optional[MoundConfig] = None,
        workspace_id: Optional[str] = None,
        event_emitter: Optional["EventEmitterProtocol"] = None,
    ) -> None:
        """
        Initialize the Knowledge Mound core.

        Args:
            config: Mound configuration. Defaults to SQLite backend.
            workspace_id: Default workspace for queries. Overrides config.
            event_emitter: Optional event emitter for cross-subsystem events.
        """
        self.config = config or MoundConfig()
        self.workspace_id = workspace_id or self.config.default_workspace_id
        self.event_emitter = event_emitter

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
        self._org_culture_manager: Optional["OrganizationCultureManager"] = None

        # State
        self._initialized = False

    # =========================================================================
    # Initialization
    # =========================================================================

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
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"PostgreSQL init failed: {e}, falling back to SQLite")
            await self._init_sqlite()
        except Exception as e:
            logger.exception(f"Unexpected PostgreSQL init error: {e}, falling back to SQLite")
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
        except (ConnectionError, TimeoutError, OSError) as e:
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
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning(f"Weaviate init failed: {e}")

    async def _init_semantic_store(self) -> None:
        """Initialize local semantic store for embeddings."""
        try:
            from aragora.knowledge.mound.semantic_store import SemanticStore

            db_path = (
                str(self.config.sqlite_path)
                if self.config.sqlite_path
                else str(DB_KNOWLEDGE_PATH / "mound.db")
            )
            # Use a separate database for semantic index
            semantic_db_path = db_path.replace(".db", "_semantic.db")
            self._semantic_store = SemanticStore(
                db_path=semantic_db_path,
                default_tenant_id=self.workspace_id,
            )
            logger.debug(f"Semantic store initialized at {semantic_db_path}")
        except ImportError:
            logger.warning("Semantic store dependencies not available")
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning(f"Semantic store init failed: {e}")

    def _ensure_initialized(self) -> None:
        """Ensure the mound is initialized."""
        if not self._initialized:
            raise RuntimeError("KnowledgeMound not initialized. Call initialize() first.")

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
            except (RuntimeError, ConnectionError, OSError) as e:
                logger.debug(f"Error closing vector store: {e}")
        if hasattr(self._meta_store, "close"):
            await self._meta_store.close()

        self._initialized = False
        logger.info("Knowledge Mound closed")

    @asynccontextmanager
    async def session(self) -> AsyncIterator["KnowledgeMoundCore"]:
        """Context manager for managed lifecycle."""
        await self.initialize()
        try:
            yield self
        finally:
            await self.close()

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self, workspace_id: Optional[str] = None) -> MoundStats:
        """Get statistics about the Knowledge Mound."""
        self._ensure_initialized()

        ws_id = workspace_id or self.workspace_id
        return await self._get_stats(ws_id)

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
    # Private Storage Adapter Methods
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
                provenance_type = source_to_provenance.get(source_type_str, ProvenanceType.DOCUMENT)
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
                cursor = conn.execute("DELETE FROM knowledge_nodes WHERE id = ?", (node_id,))
                return cursor.rowcount > 0

    async def _archive_node(self, node_id: str) -> None:
        """
        Archive node before deletion.

        Saves the node to an archive table/collection for audit trail
        and potential recovery. The archive includes full node data
        plus deletion metadata.
        """
        # Import get method from mixin - will be available via composition
        node = await self.get(node_id)  # type: ignore[attr-defined]
        if not node:
            logger.debug(f"Node {node_id} not found, skipping archive")
            return

        archive_record = {
            "id": f"arch_{node_id}_{uuid.uuid4().hex[:8]}",
            "original_id": node_id,
            "content": node.content,
            "source": node.source.value if hasattr(node.source, "value") else str(node.source),
            "source_id": node.source_id,
            "confidence": (
                node.confidence.value if hasattr(node.confidence, "value") else str(node.confidence)
            ),
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
                    conn.execute(
                        """
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
                    """
                    )
                    conn.execute(
                        """
                        INSERT INTO knowledge_archive
                        (id, original_id, content, source, source_id, confidence,
                         importance, metadata, created_at, updated_at, archived_at, workspace_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            archive_record["id"],
                            archive_record["original_id"],
                            archive_record["content"],
                            archive_record["source"],
                            archive_record["source_id"],
                            archive_record["confidence"],
                            archive_record["importance"],
                            (
                                json.dumps(archive_record["metadata"])
                                if archive_record["metadata"]
                                else "{}"
                            ),
                            archive_record["created_at"],
                            archive_record["updated_at"],
                            archive_record["archived_at"],
                            archive_record["workspace_id"],
                        ),
                    )
                    conn.commit()
                logger.debug(f"Archived node {node_id} to knowledge_archive table")
            except (RuntimeError, OSError, sqlite3.Error) as e:
                logger.warning(f"Failed to archive node {node_id}: {e}")
                # Don't block deletion on archive failure
            except Exception as e:
                logger.exception(f"Unexpected archive error for node {node_id}: {e}")
                # Don't block deletion on archive failure

    async def _save_relationship(self, from_id: str, to_id: str, rel_type: str) -> None:
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

    async def _find_by_content_hash(self, content_hash: str, workspace_id: str) -> Optional[str]:
        """Find node by content hash."""
        if hasattr(self._meta_store, "find_by_content_hash_async"):
            return await self._meta_store.find_by_content_hash_async(content_hash, workspace_id)
        else:
            node = self._meta_store.find_by_content_hash(content_hash, workspace_id)
            return node.id if node else None

    async def _increment_update_count(self, node_id: str) -> None:
        """Increment update count for a node."""
        await self._update_node(node_id, {"update_count": "update_count + 1"})

    # =========================================================================
    # Query Helper Methods (for connected stores)
    # =========================================================================

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
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Continuum query failed: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected continuum query error: {e}")
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
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Consensus query failed: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected consensus query error: {e}")
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
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Facts query failed: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected facts query error: {e}")
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
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Evidence query failed: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected evidence query error: {e}")
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
        except (KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Critique query failed: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected critique query error: {e}")
            return []

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
