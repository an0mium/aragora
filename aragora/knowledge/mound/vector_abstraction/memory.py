"""
In-memory vector store implementation.

Provides a simple in-memory vector store for testing and development.
Uses numpy for vector operations when available, falls back to pure Python.

No external dependencies required - always available.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from aragora.knowledge.mound.vector_abstraction.base import (
    BaseVectorStore,
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)

logger = logging.getLogger(__name__)

# Try to use numpy for faster vector operations
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.debug("numpy not available - using pure Python for vector operations")


@dataclass
class StoredVector:
    """A vector stored in memory."""

    id: str
    embedding: list[float]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class InMemoryVectorStore(BaseVectorStore):
    """
    In-memory vector store for testing and development.

    Stores vectors in memory with optional numpy acceleration.
    Supports all BaseVectorStore operations including hybrid search.

    Note: Data is lost when the process exits. Use for testing only.
    """

    def __init__(self, config: VectorStoreConfig):
        """Initialize in-memory store."""
        # Override backend to ensure it's MEMORY
        config.backend = VectorBackend.MEMORY
        super().__init__(config)

        # Storage: collection_name -> namespace -> id -> StoredVector
        self._collections: dict[str, dict[str, dict[str, StoredVector]]] = {}

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect (no-op for in-memory)."""
        self._connected = True
        # Ensure default collection exists
        if self.config.collection_name not in self._collections:
            await self.create_collection(self.config.collection_name)

    async def disconnect(self) -> None:
        """Disconnect (no-op for in-memory)."""
        self._connected = False

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    async def create_collection(
        self,
        name: str,
        schema: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a new collection."""
        if name not in self._collections:
            self._collections[name] = defaultdict(dict)

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if name in self._collections:
            del self._collections[name]
            return True
        return False

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        return name in self._collections

    async def list_collections(self) -> list[str]:
        """List all collections."""
        return list(self._collections.keys())

    # -------------------------------------------------------------------------
    # Data Operations
    # -------------------------------------------------------------------------

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """Insert or update a vector."""
        collection = self._get_collection()
        ns = namespace or ""

        vector = StoredVector(
            id=id,
            embedding=embedding,
            content=content,
            metadata=metadata or {},
            namespace=ns,
        )
        collection[ns][id] = vector
        return id

    async def upsert_batch(
        self,
        items: Sequence[dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> list[str]:
        """Batch upsert multiple vectors."""
        ids = []
        for item in items:
            id = await self.upsert(
                id=item.get("id") or str(uuid.uuid4()),
                embedding=item["embedding"],
                content=item["content"],
                metadata=item.get("metadata"),
                namespace=namespace,
            )
            ids.append(id)
        return ids

    async def delete(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete vectors by ID."""
        collection = self._get_collection()
        ns = namespace or ""
        deleted = 0

        for id in ids:
            if id in collection[ns]:
                del collection[ns][id]
                deleted += 1

        return deleted

    async def delete_by_filter(
        self,
        filters: dict[str, Any],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete vectors matching filter criteria."""
        collection = self._get_collection()
        ns = namespace or ""
        deleted = 0

        to_delete = []
        for id, vector in collection[ns].items():
            if self._matches_filter(vector, filters):
                to_delete.append(id)

        for id in to_delete:
            del collection[ns][id]
            deleted += 1

        return deleted

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        collection = self._get_collection()
        ns = namespace or ""

        results = []
        for vector in collection[ns].values():
            # Apply filters
            if filters and not self._matches_filter(vector, filters):
                continue

            # Calculate similarity
            score = self._cosine_similarity(embedding, vector.embedding)

            if score >= min_score:
                results.append(
                    VectorSearchResult(
                        id=vector.id,
                        content=vector.content,
                        score=score,
                        metadata=vector.metadata,
                        embedding=vector.embedding,
                    )
                )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        alpha: float = 0.5,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """
        Hybrid search combining vector and keyword matching.

        Alpha controls the balance:
        - alpha=0: Pure vector search
        - alpha=1: Pure keyword search
        - alpha=0.5: Equal weighting
        """
        collection = self._get_collection()
        ns = namespace or ""

        # Tokenize query for keyword matching
        query_tokens = set(query.lower().split())

        results = []
        for vector in collection[ns].values():
            # Apply filters
            if filters and not self._matches_filter(vector, filters):
                continue

            # Vector similarity score
            vector_score = self._cosine_similarity(embedding, vector.embedding)

            # Keyword matching score (simple BM25-like)
            content_tokens = set(vector.content.lower().split())
            if query_tokens:
                overlap = len(query_tokens & content_tokens)
                keyword_score = overlap / len(query_tokens)
            else:
                keyword_score = 0.0

            # Combined score
            combined_score = (1 - alpha) * vector_score + alpha * keyword_score

            results.append(
                VectorSearchResult(
                    id=vector.id,
                    content=vector.content,
                    score=combined_score,
                    metadata=vector.metadata,
                    embedding=vector.embedding,
                )
            )

        # Sort by combined score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    # -------------------------------------------------------------------------
    # Retrieval Operations
    # -------------------------------------------------------------------------

    async def get_by_id(
        self,
        id: str,
        namespace: Optional[str] = None,
    ) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        collection = self._get_collection()
        ns = namespace or ""

        vector = collection[ns].get(id)
        if vector:
            return VectorSearchResult(
                id=vector.id,
                content=vector.content,
                score=1.0,
                metadata=vector.metadata,
                embedding=vector.embedding,
            )
        return None

    async def get_by_ids(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """Get multiple vectors by ID."""
        results = []
        for id in ids:
            result = await self.get_by_id(id, namespace)
            if result:
                results.append(result)
        return results

    async def count(
        self,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Count vectors matching optional filters."""
        collection = self._get_collection()
        ns = namespace or ""

        if not filters:
            return len(collection[ns])

        count = 0
        for vector in collection[ns].values():
            if self._matches_filter(vector, filters):
                count += 1
        return count

    # -------------------------------------------------------------------------
    # Health & Diagnostics
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Check health and return diagnostics."""
        total_vectors = sum(
            len(ns_data)
            for collection in self._collections.values()
            for ns_data in collection.values()
        )

        return {
            "status": "healthy",
            "backend": "memory",
            "collections": len(self._collections),
            "total_vectors": total_vectors,
            "numpy_available": NUMPY_AVAILABLE,
        }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_collection(self) -> dict[str, dict[str, StoredVector]]:
        """Get the current collection, creating if needed."""
        name = self.config.collection_name
        if name not in self._collections:
            self._collections[name] = defaultdict(dict)
        return self._collections[name]

    def _matches_filter(
        self,
        vector: StoredVector,
        filters: dict[str, Any],
    ) -> bool:
        """Check if a vector matches filter criteria.

        Supports operators:
        - $eq: Equal (default when value is not a dict)
        - $ne: Not equal
        - $gt: Greater than
        - $gte: Greater than or equal
        - $lt: Less than
        - $lte: Less than or equal
        - $in: Value in list
        - $nin: Value not in list
        - $exists: Field exists (value=True) or not (value=False)
        - $contains: String contains (for text fields)

        Example:
            filters = {
                "confidence": {"$gte": 0.8},
                "node_type": {"$in": ["fact", "claim"]},
                "tags.category": "security"  # Nested + simple equality
            }
        """
        for key, value in filters.items():
            # Support nested keys with dot notation
            parts = key.split(".")
            current = vector.metadata
            field_exists = True
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    field_exists = False
                    current = None
                    break

            # Check if value is an operator dict
            if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                for op, op_value in value.items():
                    if not self._apply_operator(current, op, op_value, field_exists):
                        return False
            else:
                # Simple equality check (default $eq behavior)
                if not field_exists or current != value:
                    return False

        return True

    def _apply_operator(
        self,
        field_value: Any,
        operator: str,
        op_value: Any,
        field_exists: bool,
    ) -> bool:
        """Apply a filter operator to a field value."""
        if operator == "$eq":
            return field_exists and field_value == op_value
        elif operator == "$ne":
            return not field_exists or field_value != op_value
        elif operator == "$gt":
            return field_exists and field_value > op_value
        elif operator == "$gte":
            return field_exists and field_value >= op_value
        elif operator == "$lt":
            return field_exists and field_value < op_value
        elif operator == "$lte":
            return field_exists and field_value <= op_value
        elif operator == "$in":
            return field_exists and field_value in op_value
        elif operator == "$nin":
            return not field_exists or field_value not in op_value
        elif operator == "$exists":
            return field_exists == op_value
        elif operator == "$contains":
            return field_exists and isinstance(field_value, str) and op_value in field_value
        else:
            # Unknown operator - treat as equality for backwards compatibility
            logger.warning(f"Unknown filter operator: {operator}")
            return field_exists and field_value == op_value

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if NUMPY_AVAILABLE:
            a_np = np.array(a)
            b_np = np.array(b)
            dot = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))
        else:
            # Pure Python fallback
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))  # type: ignore[arg-type,assignment]
            norm_b = math.sqrt(sum(x * x for x in b))  # type: ignore[arg-type,assignment]
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

    # -------------------------------------------------------------------------
    # Testing Utilities
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        self._collections.clear()

    def get_all_vectors(
        self,
        namespace: Optional[str] = None,
    ) -> list[StoredVector]:
        """Get all vectors in a namespace (for testing)."""
        collection = self._get_collection()
        ns = namespace or ""
        return list(collection[ns].values())
