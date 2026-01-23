"""
Qdrant vector store adapter.

Provides integration with Qdrant vector database for the Knowledge Mound.

Requirements:
    pip install qdrant-client>=1.6
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from typing import Any, Optional, Sequence

from aragora.knowledge.mound.vector_abstraction.base import (
    BaseVectorStore,
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)

logger = logging.getLogger(__name__)

# Check for qdrant library
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient  # noqa: F401
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.models import (
        Distance,
        PointStruct,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.debug("qdrant-client not available - install with: pip install qdrant-client")


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant vector store adapter.

    Provides high-performance vector storage with:
    - Payload filtering
    - Scalar and product quantization
    - Snapshots and backups
    - Multi-tenant support via collection partitions

    Usage:
        config = VectorStoreConfig(
            backend=VectorBackend.QDRANT,
            url="http://localhost:6333",
            collection_name="knowledge_mound",
        )
        store = QdrantVectorStore(config)
        await store.connect()
    """

    def __init__(self, config: VectorStoreConfig):
        """Initialize Qdrant store."""
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant client not installed. Install with: pip install qdrant-client"
            )

        config.backend = VectorBackend.QDRANT
        super().__init__(config)

        self._client: Optional[AsyncQdrantClient] = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to Qdrant."""
        if self._connected:
            return

        try:
            url = self.config.url or "http://localhost:6333"

            # Create async client
            if self.config.api_key:
                self._client = AsyncQdrantClient(
                    url=url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout_seconds,
                )
            else:
                self._client = AsyncQdrantClient(
                    url=url,
                    timeout=self.config.timeout_seconds,
                )

            self._connected = True

            # Ensure default collection exists
            if not await self.collection_exists(self.config.collection_name):
                await self.create_collection(self.config.collection_name)

            logger.info(f"Connected to Qdrant at {url}")

        except (ConnectionError, TimeoutError, OSError) as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Qdrant: {e}") from e
        except Exception as e:
            self._connected = False
            logger.exception(f"Unexpected Qdrant connection error: {e}")
            raise ConnectionError(f"Failed to connect to Qdrant: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to Qdrant."""
        if self._client:
            try:
                await self._client.close()
            except (RuntimeError, ConnectionError, OSError) as e:
                logger.warning(f"Error closing Qdrant connection: {e}")
            finally:
                self._client = None
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
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        # Check if exists
        collections = await self._client.get_collections()
        if any(c.name == name for c in collections.collections):
            return

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot_product": Distance.DOT,
        }
        distance = distance_map.get(self.config.distance_metric, Distance.COSINE)

        # Create collection
        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self.config.embedding_dimensions,
                distance=distance,
            ),
        )

        logger.info(f"Created Qdrant collection: {name}")

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        try:
            await self._client.delete_collection(name)
            return True
        except (RuntimeError, ConnectionError, KeyError) as e:
            logger.debug(f"Delete collection failed: {e}")
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        collections = await self._client.get_collections()
        return any(c.name == name for c in collections.collections)

    async def list_collections(self) -> list[str]:
        """List all collections."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        collections = await self._client.get_collections()
        return [c.name for c in collections.collections]

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
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        payload = {
            "content": content,
            "namespace": namespace or "",
            **(metadata or {}),
        }

        # Convert string ID to UUID for Qdrant
        point_id = self._to_qdrant_id(id)

        await self._client.upsert(
            collection_name=self.config.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )

        return id

    async def upsert_batch(
        self,
        items: Sequence[dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> list[str]:
        """Batch upsert multiple vectors."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        points = []
        ids = []

        for item in items:
            item_id = item.get("id") or str(uuid_lib.uuid4())
            payload = {
                "content": item["content"],
                "namespace": namespace or "",
                **(item.get("metadata") or {}),
            }

            points.append(
                PointStruct(
                    id=self._to_qdrant_id(item_id),
                    vector=item["embedding"],
                    payload=payload,
                )
            )
            ids.append(item_id)

        # Batch upsert
        await self._client.upsert(
            collection_name=self.config.collection_name,
            points=points,
        )

        return ids

    async def delete(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete vectors by ID."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        point_ids = [self._to_qdrant_id(id) for id in ids]

        await self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=qdrant_models.PointIdsList(points=point_ids),
        )

        return len(ids)

    async def delete_by_filter(
        self,
        filters: dict[str, Any],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete vectors matching filter criteria."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        qdrant_filter = self._build_filter(filters, namespace)

        # Count before delete
        count_before = await self.count(filters, namespace)

        await self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=qdrant_models.FilterSelector(filter=qdrant_filter),
        )

        return count_before

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
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        qdrant_filter = self._build_filter(filters, namespace)

        response = await self._client.search(
            collection_name=self.config.collection_name,
            query_vector=embedding,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=min_score,
            with_payload=True,
            with_vectors=True,
        )

        results = []
        for point in response:
            payload = point.payload or {}
            results.append(
                VectorSearchResult(
                    id=self._from_qdrant_id(point.id),
                    content=payload.get("content", ""),
                    score=point.score,
                    metadata={
                        k: v for k, v in payload.items() if k not in ["content", "namespace"]
                    },
                    embedding=point.vector if isinstance(point.vector, list) else None,
                )
            )

        return results

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
        Hybrid search combining vector and text matching.

        Note: Qdrant doesn't have built-in BM25, so we approximate
        by boosting results that contain query terms.
        """
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        # Get more results for re-ranking
        vector_results = await self.search(
            embedding=embedding,
            limit=limit * 2,
            filters=filters,
            namespace=namespace,
        )

        # Re-rank with keyword boost
        query_tokens = set(query.lower().split())

        reranked = []
        for result in vector_results:
            content_tokens = set(result.content.lower().split())

            # Calculate keyword overlap
            if query_tokens:
                keyword_score = len(query_tokens & content_tokens) / len(query_tokens)
            else:
                keyword_score = 0.0

            # Combined score
            combined_score = (1 - alpha) * result.score + alpha * keyword_score

            reranked.append(
                VectorSearchResult(
                    id=result.id,
                    content=result.content,
                    score=combined_score,
                    metadata=result.metadata,
                    embedding=result.embedding,
                )
            )

        # Sort by combined score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:limit]

    # -------------------------------------------------------------------------
    # Retrieval Operations
    # -------------------------------------------------------------------------

    async def get_by_id(
        self,
        id: str,
        namespace: Optional[str] = None,
    ) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        try:
            points = await self._client.retrieve(
                collection_name=self.config.collection_name,
                ids=[self._to_qdrant_id(id)],
                with_payload=True,
                with_vectors=True,
            )

            if points:
                point = points[0]
                payload = point.payload or {}
                return VectorSearchResult(
                    id=self._from_qdrant_id(point.id),
                    content=payload.get("content", ""),
                    score=1.0,
                    metadata={
                        k: v for k, v in payload.items() if k not in ["content", "namespace"]
                    },
                    embedding=point.vector if isinstance(point.vector, list) else None,
                )
        except (RuntimeError, ConnectionError, KeyError) as e:
            logger.debug(f"Error retrieving vector by ID: {e}")

        return None

    async def get_by_ids(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """Get multiple vectors by ID."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        point_ids = [self._to_qdrant_id(id) for id in ids]

        points = await self._client.retrieve(
            collection_name=self.config.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )

        results = []
        for point in points:
            payload = point.payload or {}
            results.append(
                VectorSearchResult(
                    id=self._from_qdrant_id(point.id),
                    content=payload.get("content", ""),
                    score=1.0,
                    metadata={
                        k: v for k, v in payload.items() if k not in ["content", "namespace"]
                    },
                    embedding=point.vector if isinstance(point.vector, list) else None,
                )
            )

        return results

    async def count(
        self,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Count vectors matching optional filters."""
        if not self._client:
            raise ConnectionError("Not connected to Qdrant")

        qdrant_filter = self._build_filter(filters, namespace)

        result = await self._client.count(
            collection_name=self.config.collection_name,
            count_filter=qdrant_filter,
        )

        return result.count

    # -------------------------------------------------------------------------
    # Health & Diagnostics
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Check health and return diagnostics."""
        if not self._client:
            return {"status": "disconnected", "backend": "qdrant"}

        try:
            # Get cluster info
            info = await self._client.get_collections()
            return {
                "status": "healthy",
                "backend": "qdrant",
                "collections": len(info.collections),
            }
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            return {
                "status": "unhealthy",
                "backend": "qdrant",
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _to_qdrant_id(self, id: str) -> str | int:
        """Convert string ID to Qdrant-compatible ID."""
        # Try to use as UUID, fall back to hash
        try:
            return str(uuid_lib.UUID(id))
        except ValueError:
            # Use hash for non-UUID strings
            return id

    def _from_qdrant_id(self, id: Any) -> str:
        """Convert Qdrant ID back to string."""
        return str(id)

    def _build_filter(
        self,
        filters: Optional[dict[str, Any]],
        namespace: Optional[str],
    ) -> Optional[Filter]:
        """Build Qdrant filter from dict."""
        conditions = []

        if namespace:
            conditions.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))

        if filters:
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        if not conditions:
            return None

        return Filter(must=conditions)
