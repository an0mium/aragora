"""
Knowledge Mound - Unified knowledge storage with vector + graph capabilities.

Implements the "termite mound" architecture where agents contribute to a shared
superstructure of knowledge. Unifies ContinuumMemory, ConsensusMemory, and FactStore
into a coherent knowledge graph with semantic search.

Key concepts:
- KnowledgeNode: A unit of knowledge (fact, claim, memory, evidence, consensus)
- ProvenanceChain: Tracks origin and transformations of knowledge
- Graph relationships: supports, contradicts, derived_from for knowledge traversal

Usage:
    from aragora.knowledge.mound import KnowledgeMound, KnowledgeNode

    mound = KnowledgeMound(workspace_id="ws_123")
    await mound.initialize()

    # Add knowledge
    node = KnowledgeNode(
        node_type="fact",
        content="API keys should never be committed to version control",
        confidence=0.95,
    )
    node_id = await mound.add_node(node)

    # Query semantically
    results = await mound.query_semantic("security best practices", limit=10)

    # Traverse graph
    related = await mound.query_graph(node_id, "supports", depth=2)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from aragora.knowledge.types import ValidationStatus
from aragora.memory.tier_manager import MemoryTier

# Import types from extracted module
from .mound_types import (
    KnowledgeNode,
    KnowledgeQueryResult,
    KnowledgeRelationship,
    NodeType,
    ProvenanceChain,
    ProvenanceType,
    RelationshipType,
    _to_enum_value,
    _to_iso_string,
)

# Import store from extracted module
from .mound_store import KnowledgeMoundMetaStore

logger = logging.getLogger(__name__)


class KnowledgeMound:
    """
    Unified knowledge storage with vector + graph capabilities.

    Combines:
    - SQLite for metadata, relationships, fast queries
    - Weaviate for vector embeddings and semantic search
    - Graph traversal for knowledge inference

    This is the "termite mound" - a shared superstructure where agents
    contribute knowledge that accumulates across tasks and sessions.
    """

    def __init__(
        self,
        workspace_id: str = "default",
        db_path: str | Path | None = None,
        weaviate_config: dict[str, Any] | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        """
        Initialize the Knowledge Mound.

        Args:
            workspace_id: Workspace for multi-tenant isolation
            db_path: Path to SQLite database
            weaviate_config: Configuration for Weaviate vector store
            embedding_fn: Function to generate embeddings (defaults to internal)
        """
        self.workspace_id = workspace_id
        if db_path is None:
            # Use the nomic directory for knowledge mound storage
            from aragora.persistence.db_config import get_nomic_dir

            knowledge_dir = get_nomic_dir() / "knowledge"
            self._db_path = knowledge_dir / "mound.db"
        else:
            self._db_path = Path(db_path)
        self._weaviate_config = weaviate_config
        self._embedding_fn = embedding_fn

        # Initialize stores
        self._meta_store: KnowledgeMoundMetaStore | None = None
        self._vector_store: Any | None = None  # WeaviateStore when available
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the knowledge mound stores."""
        if self._initialized:
            return

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite meta store
        self._meta_store = KnowledgeMoundMetaStore(self._db_path)

        # Try to initialize Weaviate if config provided
        if self._weaviate_config:
            try:
                from aragora.documents.indexing.weaviate_store import WeaviateStore, WeaviateConfig

                config = WeaviateConfig(**self._weaviate_config)
                self._vector_store = WeaviateStore(config)
                await self._vector_store.connect()
                logger.info("Knowledge Mound initialized with Weaviate vector store")
            except ImportError:
                logger.warning("Weaviate not available - using SQLite-only mode")
            except Exception as e:
                logger.warning(f"Failed to connect to Weaviate: {e} - using SQLite-only mode")

        self._initialized = True
        logger.info(f"Knowledge Mound initialized for workspace: {self.workspace_id}")

    def _ensure_initialized(self) -> None:
        """Ensure the mound is initialized."""
        if not self._initialized or not self._meta_store:
            raise RuntimeError("KnowledgeMound not initialized. Call initialize() first.")

    async def add_node(
        self,
        node: KnowledgeNode,
        deduplicate: bool = True,
    ) -> str:
        """
        Add a knowledge node to the mound.

        Args:
            node: The knowledge node to add
            deduplicate: If True, check for existing node with same content

        Returns:
            The node ID
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        # Set workspace if not set
        if not node.workspace_id:
            node.workspace_id = self.workspace_id

        # Check for duplicates
        if deduplicate:
            existing = self._meta_store.find_by_content_hash(node.content_hash, node.workspace_id)
            if existing:
                # Update existing node
                existing.update_count += 1
                existing.updated_at = datetime.now()
                # Merge confidence (weighted average)
                existing.confidence = existing.confidence * 0.7 + node.confidence * 0.3
                node = existing

        # Save to SQLite
        self._meta_store.save_node(node)

        # Save embedding to Weaviate when vector store is available
        if self._vector_store and self._embedding_fn:
            try:
                embedding = self._embedding_fn(node.content)
                await self._vector_store.upsert(
                    id=node.id,
                    embedding=embedding,
                    content=node.content,
                    metadata={
                        "node_type": node.node_type,
                        "confidence": node.confidence,
                        "workspace_id": node.workspace_id,
                    },
                    namespace=node.workspace_id,
                )
            except Exception as e:
                logger.warning(f"Failed to save embedding to vector store: {e}")

        logger.debug(f"Added knowledge node: {node.id} ({node.node_type})")
        return node.id

    async def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Get a knowledge node by ID."""
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")
        return self._meta_store.get_node(node_id)

    async def add_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: RelationshipType,
        strength: float = 1.0,
        created_by: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a relationship between two nodes.

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Type of relationship
            strength: Relationship strength (0-1)
            created_by: Agent/user who created relationship
            metadata: Additional metadata

        Returns:
            Relationship ID
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        rel = KnowledgeRelationship(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship_type=relationship_type,
            strength=strength,
            created_by=created_by,
            metadata=metadata or {},
        )

        return self._meta_store.save_relationship(rel)

    async def query_semantic(
        self,
        query: str,
        limit: int = 10,
        node_types: list[NodeType] | None = None,
        min_confidence: float = 0.0,
        workspace_id: str | None = None,
    ) -> KnowledgeQueryResult:
        """
        Semantic search across the knowledge mound.

        Args:
            query: Natural language query
            limit: Maximum results
            node_types: Filter by node types
            min_confidence: Minimum confidence threshold
            workspace_id: Filter by workspace (defaults to self.workspace_id)

        Returns:
            Query result with matching nodes
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        import time

        start = time.time()

        ws_id = workspace_id or self.workspace_id

        # Use Weaviate for semantic search when available
        if self._vector_store and self._embedding_fn:
            try:
                query_embedding = self._embedding_fn(query)
                if node_types:
                    # Note: Weaviate filters multiple types via OR, we'll filter post-search
                    pass

                vector_results = await self._vector_store.search(
                    embedding=query_embedding,
                    limit=limit * 2,  # Fetch extra for filtering
                    namespace=ws_id,
                    min_score=min_confidence,
                )

                # Get full nodes from SQLite and filter by type/confidence
                result_nodes = []
                for vr in vector_results:
                    node = self._meta_store.get_node(vr.id)
                    if node:
                        if node_types and node.node_type not in node_types:
                            continue
                        if node.confidence < min_confidence:
                            continue
                        result_nodes.append(node)
                        if len(result_nodes) >= limit:
                            break

                elapsed_ms = int((time.time() - start) * 1000)
                return KnowledgeQueryResult(
                    nodes=result_nodes,
                    total_count=len(result_nodes),
                    query=query,
                    processing_time_ms=elapsed_ms,
                )
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keyword: {e}")

        # Fall back to keyword-based search
        nodes = self._meta_store.query_nodes(
            workspace_id=ws_id,
            node_types=node_types,
            min_confidence=min_confidence,
            limit=limit,
        )

        # Simple keyword relevance scoring
        query_words = set(query.lower().split())
        scored_nodes = []
        for node in nodes:
            content_words = set(node.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1)
            scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        result_nodes = [node for _, node in scored_nodes[:limit]]

        elapsed_ms = int((time.time() - start) * 1000)

        return KnowledgeQueryResult(
            nodes=result_nodes,
            total_count=len(result_nodes),
            query=query,
            processing_time_ms=elapsed_ms,
        )

    async def query_graph(
        self,
        start_node_id: str,
        relationship_type: RelationshipType | None = None,
        depth: int = 2,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    ) -> list[KnowledgeNode]:
        """
        Graph traversal from a starting node.

        Args:
            start_node_id: Starting node ID
            relationship_type: Filter by relationship type
            depth: Maximum traversal depth
            direction: Direction of traversal

        Returns:
            List of connected nodes
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        visited: set[str] = set()
        result: list[KnowledgeNode] = []

        async def traverse(node_id: str, current_depth: int) -> None:
            if current_depth > depth or node_id in visited:
                return

            visited.add(node_id)
            node = self._meta_store.get_node(node_id)
            if node:
                result.append(node)

            relationships = self._meta_store.get_relationships(
                node_id, relationship_type, direction
            )

            for rel in relationships:
                next_id = rel.to_node_id if direction != "incoming" else rel.from_node_id
                if next_id != node_id:
                    await traverse(next_id, current_depth + 1)

        await traverse(start_node_id, 0)
        return result

    async def query_nodes(
        self,
        node_types: list[NodeType] | None = None,
        min_confidence: float = 0.0,
        tier: MemoryTier | None = None,
        validation_status: ValidationStatus | None = None,
        topics: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
        workspace_id: str | None = None,
    ) -> list[KnowledgeNode]:
        """
        Query nodes with filters.

        Args:
            node_types: Filter by node types
            min_confidence: Minimum confidence
            tier: Filter by memory tier
            validation_status: Filter by validation status
            topics: Filter by topics
            limit: Maximum results
            offset: Skip first N results
            workspace_id: Filter by workspace

        Returns:
            List of matching nodes
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        return self._meta_store.query_nodes(
            workspace_id=workspace_id or self.workspace_id,
            node_types=node_types,
            min_confidence=min_confidence,
            tier=tier,
            validation_status=validation_status,
            topics=topics,
            limit=limit,
            offset=offset,
        )

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge mound."""
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")
        return self._meta_store.get_stats(self.workspace_id)

    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a knowledge node and its relationships.

        Args:
            node_id: The node ID to delete

        Returns:
            True if deleted, False if node not found
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        # Delete from vector store if available
        if self._vector_store:
            try:
                await self._vector_store.delete(node_id)
            except Exception as e:
                logger.warning(f"Failed to delete node from vector store: {e}")

        return self._meta_store.delete_node(node_id)

    async def query_by_provenance(
        self,
        source_type: str | None = None,
        source_id: str | None = None,
        node_type: str | None = None,
        limit: int = 100,
        workspace_id: str | None = None,
    ) -> list[KnowledgeNode]:
        """
        Query nodes by provenance attributes.

        Args:
            source_type: Filter by provenance source type (e.g., "workflow_engine", "debate")
            source_id: Filter by provenance source ID (e.g., workflow_id, debate_id)
            node_type: Filter by node type
            limit: Maximum results to return
            workspace_id: Filter by workspace (defaults to self.workspace_id)

        Returns:
            List of matching KnowledgeNodes
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        ws_id = workspace_id or self.workspace_id
        node_ids = self._meta_store.query_by_provenance(
            source_type=source_type,
            source_id=source_id,
            node_type=node_type,
            workspace_id=ws_id,
            limit=limit,
        )

        nodes = []
        for node_id in node_ids:
            node = self._meta_store.get_node(node_id)
            if node:
                nodes.append(node)

        return nodes

    async def get_relationships(
        self,
        node_id: str,
        relationship_type: RelationshipType | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "both",
    ) -> list[KnowledgeRelationship]:
        """
        Get relationships for a specific node.

        Args:
            node_id: The node ID to get relationships for
            relationship_type: Filter by relationship type (optional)
            direction: Direction of relationships ('outgoing', 'incoming', or 'both')

        Returns:
            List of relationships
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")
        return self._meta_store.get_relationships(node_id, relationship_type, direction)

    async def merge_from_debate(
        self,
        debate_result: Any,  # DebateResult type
        extract_facts: bool = True,
    ) -> list[str]:
        """
        Extract and store knowledge from a debate outcome.

        Args:
            debate_result: Result from Arena.run()
            extract_facts: Whether to extract facts from messages

        Returns:
            List of created node IDs
        """
        self._ensure_initialized()
        created_ids: list[str] = []

        # Create consensus node
        if hasattr(debate_result, "consensus") and debate_result.consensus:
            consensus_node = KnowledgeNode(
                node_type="consensus",
                content=debate_result.consensus,
                confidence=getattr(debate_result, "confidence", 0.8),
                provenance=ProvenanceChain(
                    source_type=ProvenanceType.DEBATE,
                    source_id=getattr(debate_result, "debate_id", ""),
                    debate_id=getattr(debate_result, "debate_id", None),
                ),
                workspace_id=self.workspace_id,
                validation_status=ValidationStatus.MAJORITY_AGREED,
            )
            node_id = await self.add_node(consensus_node)
            created_ids.append(node_id)

        # Extract facts from debate messages when extract_facts=True
        if extract_facts and hasattr(debate_result, "messages"):
            messages = debate_result.messages
            for msg in messages:
                # Extract factual claims from agent messages
                content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
                agent_id = getattr(msg, "agent", "unknown") if hasattr(msg, "agent") else "unknown"

                # Skip short or non-substantive messages
                if len(content) < 50:
                    continue

                # Create a claim node from each substantive message
                claim_node = KnowledgeNode(
                    node_type="claim",
                    content=content[:2000],  # Truncate long content
                    confidence=0.6,  # Lower confidence for unverified claims
                    provenance=ProvenanceChain(
                        source_type=ProvenanceType.AGENT,
                        source_id=agent_id,
                        agent_id=agent_id,
                        debate_id=getattr(debate_result, "debate_id", None),
                    ),
                    workspace_id=self.workspace_id,
                    validation_status=ValidationStatus.UNVERIFIED,
                    metadata={
                        "agent": agent_id,
                        "debate_round": getattr(msg, "round", 0),
                    },
                )
                claim_id = await self.add_node(claim_node)
                created_ids.append(claim_id)

                # Link claim to consensus if one exists
                if created_ids and created_ids[0] != claim_id:
                    await self.add_relationship(
                        from_node_id=claim_id,
                        to_node_id=created_ids[0],  # Consensus node
                        relationship_type="supports",
                        strength=0.5,
                        created_by=agent_id,
                    )

        return created_ids

    async def export_graph_d3(
        self,
        start_node_id: str | None = None,
        depth: int = 3,
        limit: int = 100,
    ) -> dict[str, Any]:
        """
        Export graph in D3.js-compatible format.

        Args:
            start_node_id: Starting node for traversal (None for all nodes)
            depth: Maximum traversal depth
            limit: Maximum number of nodes

        Returns:
            Dict with 'nodes' and 'links' arrays for D3 force-directed graph
        """
        self._ensure_initialized()
        if self._meta_store is None:
            raise RuntimeError("Meta store not initialized - call initialize() first")

        nodes: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        node_ids: set[str] = set()

        if start_node_id:
            # Traverse from starting node
            traversed = await self.query_graph(start_node_id, depth=depth, direction="both")
            for node in traversed[:limit]:
                if node.id not in node_ids:
                    node_ids.add(node.id)
                    nodes.append(
                        {
                            "id": node.id,
                            "label": node.content[:100] if node.content else "",
                            "type": node.node_type,
                            "confidence": node.confidence,
                            "tier": node.tier.value if node.tier else "medium",
                            "validation": (
                                node.validation_status.value
                                if node.validation_status
                                else "pending"
                            ),
                        }
                    )
        else:
            # Get all nodes up to limit
            all_nodes = await self.query_nodes(limit=limit)
            for node in all_nodes:
                node_ids.add(node.id)
                nodes.append(
                    {
                        "id": node.id,
                        "label": node.content[:100] if node.content else "",
                        "type": node.node_type,
                        "confidence": node.confidence,
                        "tier": node.tier.value if node.tier else "medium",
                        "validation": (
                            node.validation_status.value if node.validation_status else "pending"
                        ),
                    }
                )

        # Get relationships between collected nodes
        for node_id in node_ids:
            rels = self._meta_store.get_relationships(node_id, direction="outgoing")
            for rel in rels:
                if rel.to_node_id in node_ids:
                    links.append(
                        {
                            "source": rel.from_node_id,
                            "target": rel.to_node_id,
                            "type": rel.relationship_type,
                            "strength": rel.strength,
                        }
                    )

        return {"nodes": nodes, "links": links}

    async def export_graph_graphml(
        self,
        start_node_id: str | None = None,
        depth: int = 3,
        limit: int = 100,
    ) -> str:
        """
        Export graph in GraphML format.

        Args:
            start_node_id: Starting node for traversal (None for all nodes)
            depth: Maximum traversal depth
            limit: Maximum number of nodes

        Returns:
            GraphML XML string
        """
        d3_data = await self.export_graph_d3(start_node_id, depth, limit)

        # Build GraphML XML
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
            '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '  <key id="rel_type" for="edge" attr.name="type" attr.type="string"/>',
            '  <key id="strength" for="edge" attr.name="strength" attr.type="double"/>',
            '  <graph id="knowledge_graph" edgedefault="directed">',
        ]

        # Add nodes
        for node in d3_data["nodes"]:
            # Escape XML special characters in label
            label = (
                (node.get("label", "") or "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            lines.append(f'    <node id="{node["id"]}">')
            lines.append(f'      <data key="label">{label}</data>')
            lines.append(f'      <data key="type">{node.get("type", "unknown")}</data>')
            lines.append(f'      <data key="confidence">{node.get("confidence", 0.0)}</data>')
            lines.append("    </node>")

        # Add edges
        for i, link in enumerate(d3_data["links"]):
            lines.append(
                f'    <edge id="e{i}" source="{link["source"]}" target="{link["target"]}">'
            )
            lines.append(f'      <data key="rel_type">{link.get("type", "related")}</data>')
            lines.append(f'      <data key="strength">{link.get("strength", 0.5)}</data>')
            lines.append("    </edge>")

        lines.append("  </graph>")
        lines.append("</graphml>")

        return "\n".join(lines)

    async def close(self) -> None:
        """Close connections."""
        if self._vector_store:
            try:
                await self._vector_store.close()
            except Exception as e:
                logger.debug(f"Error closing vector store: {e}")
        self._initialized = False


# Re-export types for backwards compatibility
__all__ = [
    "KnowledgeMound",
    "KnowledgeMoundMetaStore",
    "KnowledgeNode",
    "KnowledgeRelationship",
    "KnowledgeQueryResult",
    "ProvenanceChain",
    "ProvenanceType",
    "NodeType",
    "RelationshipType",
]
