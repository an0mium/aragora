"""
CRUD Operations Mixin for Knowledge Mound.

Provides core create, read, update, delete operations:
- store: Store with provenance tracking and deduplication
- get: Retrieve by ID with caching
- update: Update node fields
- delete: Delete with optional archival
- add: Simplified content addition
- add_node: KnowledgeNode adapter
- get_node: KnowledgeNode retrieval adapter
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol

if TYPE_CHECKING:
    from aragora.knowledge.mound.types import (
        IngestionRequest,
        IngestionResult,
        KnowledgeItem,
        KnowledgeSource,
        MoundConfig,
    )

logger = logging.getLogger(__name__)


class CRUDProtocol(Protocol):
    """Protocol defining expected interface for CRUD mixin."""

    config: "MoundConfig"
    workspace_id: str
    _cache: Optional[Any]
    _semantic_store: Optional[Any]
    _initialized: bool
    event_emitter: Optional[Any]

    def _ensure_initialized(self) -> None: ...
    async def _save_node(self, node_data: Dict[str, Any]) -> None: ...
    async def _get_node(self, node_id: str) -> Optional["KnowledgeItem"]: ...
    async def _update_node(self, node_id: str, updates: Dict[str, Any]) -> None: ...
    async def _delete_node(self, node_id: str) -> bool: ...
    async def _archive_node(self, node_id: str) -> None: ...
    async def _save_relationship(self, from_id: str, to_id: str, rel_type: str) -> None: ...
    async def _find_by_content_hash(self, content_hash: str, workspace_id: str) -> Optional[str]: ...
    async def _increment_update_count(self, node_id: str) -> None: ...


class CRUDOperationsMixin:
    """Mixin providing CRUD operations for KnowledgeMound."""

    async def store(self: CRUDProtocol, request: "IngestionRequest") -> "IngestionResult":
        """
        Store a new knowledge item with full provenance tracking.

        Args:
            request: Ingestion request with content and metadata

        Returns:
            IngestionResult with node ID and status
        """
        from aragora.knowledge.mound.types import IngestionResult

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

        # Emit KNOWLEDGE_INDEXED event for cross-subsystem tracking
        if self.event_emitter:
            try:
                from aragora.events.types import StreamEvent, StreamEventType

                self.event_emitter.emit(
                    StreamEvent(
                        type=StreamEventType.KNOWLEDGE_INDEXED,
                        data={
                            "node_id": node_id,
                            "content": request.content[:200],
                            "node_type": request.node_type,
                            "workspace_id": request.workspace_id,
                            "source_type": request.source_type.value,
                            "confidence": request.confidence,
                        },
                    )
                )
            except (ImportError, AttributeError, TypeError):
                pass  # Events module not available

        return IngestionResult(
            node_id=node_id,
            success=True,
            relationships_created=relationships_created,
        )

    async def get(self: CRUDProtocol, node_id: str) -> Optional["KnowledgeItem"]:
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

    async def update(
        self: CRUDProtocol, node_id: str, updates: Dict[str, Any]
    ) -> Optional["KnowledgeItem"]:
        """Update a knowledge node."""
        self._ensure_initialized()

        updates["updated_at"] = datetime.now()
        await self._update_node(node_id, updates)

        # Invalidate cache
        if self._cache:
            await self._cache.invalidate_node(node_id)

        return await self.get(node_id)

    async def delete(self: CRUDProtocol, node_id: str, archive: bool = True) -> bool:
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
        self: CRUDProtocol,
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
        from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

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

    async def add_node(self: CRUDProtocol, node: Any) -> str:
        """
        Add a KnowledgeNode to the mound.

        This is an adapter method for compatibility with checkpoint_store
        and other components that use the KnowledgeNode interface.

        Args:
            node: KnowledgeNode with node_type, content, confidence, etc.

        Returns:
            The created node ID
        """
        from aragora.knowledge.mound_core import KnowledgeNode

        if not isinstance(node, KnowledgeNode):
            raise TypeError(f"Expected KnowledgeNode, got {type(node)}")

        # Convert tier to string if it's an enum
        tier_str = node.tier.value if hasattr(node.tier, "value") else str(node.tier)

        return await self.add(
            content=node.content,
            metadata={
                "provenance": node.provenance.__dict__ if node.provenance else None,
                "node_type": node.node_type,
            },
            workspace_id=node.workspace_id or self.workspace_id,
            node_type=node.node_type if isinstance(node.node_type, str) else node.node_type,
            confidence=node.confidence,
            tier=tier_str,
        )

    async def get_node(self: CRUDProtocol, node_id: str) -> Optional[Any]:
        """
        Get a KnowledgeNode by ID.

        This is an adapter method for compatibility with checkpoint_store
        and other components that use the KnowledgeNode interface.

        Args:
            node_id: The node ID to retrieve

        Returns:
            A dict-like object with node_type, content, etc., or None
        """
        from aragora.knowledge.mound.types import KnowledgeItem

        item = await self.get(node_id)
        if item is None:
            return None

        # Return a dict-like object that has node_type and content
        # This maintains compatibility with code expecting KnowledgeNode interface
        class NodeProxy:
            def __init__(self, item: KnowledgeItem) -> None:
                self.id = item.id
                self.content = item.content
                # ConfidenceLevel is a string enum (e.g., "medium"), map to numeric
                if hasattr(item.confidence, "value"):
                    conf_val = item.confidence.value
                    # Map string confidence levels to numeric values
                    conf_map = {"verified": 0.95, "high": 0.85, "medium": 0.7, "low": 0.4, "unverified": 0.2}
                    self.confidence = conf_map.get(conf_val, 0.5)
                elif isinstance(item.confidence, (int, float)):
                    self.confidence = float(item.confidence)
                else:
                    self.confidence = 0.5
                self.node_type = item.metadata.get("node_type", "fact")

        return NodeProxy(item)
