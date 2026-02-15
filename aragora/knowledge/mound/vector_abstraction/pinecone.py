"""
Pinecone vector store adapter.

Provides integration with Pinecone vector database for the Knowledge Mound.

Requirements:
    pip install pinecone-client>=3.0
"""

from __future__ import annotations

import logging
import os
from typing import Any
from collections.abc import Sequence

from aragora.knowledge.mound.vector_abstraction.base import (
    BaseVectorStore,
    VectorBackend,
    VectorSearchResult,
    VectorStoreConfig,
)

logger = logging.getLogger(__name__)

# Check for pinecone library
try:
    from pinecone import Pinecone, ServerlessSpec, PodSpec

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None
    ServerlessSpec = None
    PodSpec = None
    logger.debug("pinecone-client not available - install with: pip install pinecone-client")


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone vector store adapter.

    Provides managed vector storage with:
    - Serverless and pod-based deployment options
    - Metadata filtering
    - Namespace-based multi-tenancy
    - Automatic scaling
    - Global distribution

    Configuration:
        PINECONE_API_KEY: API key for authentication
        PINECONE_ENVIRONMENT: Environment (e.g., "us-east-1-aws")
        PINECONE_INDEX: Index name

    Usage:
        config = VectorStoreConfig(
            backend=VectorBackend.PINECONE,
            api_key=os.getenv("PINECONE_API_KEY"),
            collection_name="knowledge-mound",
            extra={"environment": "us-east-1-aws"},
        )
        store = PineconeVectorStore(config)
        await store.connect()
    """

    def __init__(self, config: VectorStoreConfig):
        """Initialize Pinecone store."""
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "Pinecone client not installed. Install with: pip install pinecone-client"
            )

        config.backend = VectorBackend.PINECONE
        super().__init__(config)

        self._client: Pinecone | None = None
        self._index = None
        self._index_name = config.collection_name

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to Pinecone."""
        if self._connected:
            return

        try:
            api_key = self.config.api_key or os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Pinecone API key not configured")

            self._client = Pinecone(api_key=api_key)

            # Check if index exists
            existing_indexes = self._client.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes]

            if self._index_name not in index_names:
                # Create serverless index by default
                environment = self.config.extra.get(
                    "environment",
                    os.environ.get("PINECONE_ENVIRONMENT", "us-east-1"),
                )
                cloud = self.config.extra.get("cloud", "aws")

                self._client.create_index(
                    name=self._index_name,
                    dimension=self.config.embedding_dimensions,
                    metric=self._map_metric(self.config.distance_metric),
                    spec=ServerlessSpec(cloud=cloud, region=environment),
                )
                logger.info(
                    f"Created Pinecone serverless index: {self._index_name} "
                    f"in {cloud}/{environment}"
                )

            self._index = self._client.Index(self._index_name)
            self._connected = True
            logger.info(f"Connected to Pinecone index: {self._index_name}")

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Pinecone: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to Pinecone."""
        # Pinecone client doesn't require explicit disconnect
        self._client = None
        self._index = None
        self._connected = False

    def _map_metric(self, metric: str) -> str:
        """Map generic metric name to Pinecone metric."""
        metric_map = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "dot_product": "dotproduct",
        }
        return metric_map.get(metric, "cosine")

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    async def create_collection(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Create a new index. In Pinecone, collections are namespaces within an index."""
        if not self._client:
            raise ConnectionError("Not connected to Pinecone")

        # Check if already exists
        existing_indexes = self._client.list_indexes()
        index_names = [idx.name for idx in existing_indexes.indexes]

        if name in index_names:
            return

        environment = self.config.extra.get(
            "environment",
            os.environ.get("PINECONE_ENVIRONMENT", "us-east-1"),
        )
        cloud = self.config.extra.get("cloud", "aws")

        self._client.create_index(
            name=name,
            dimension=self.config.embedding_dimensions,
            metric=self._map_metric(self.config.distance_metric),
            spec=ServerlessSpec(cloud=cloud, region=environment),
        )
        logger.info(f"Created Pinecone index: {name}")

    async def delete_collection(self, name: str) -> bool:
        """Delete an index."""
        if not self._client:
            raise ConnectionError("Not connected to Pinecone")

        try:
            self._client.delete_index(name)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete Pinecone index {name}: {e}")
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if an index exists."""
        if not self._client:
            raise ConnectionError("Not connected to Pinecone")

        existing_indexes = self._client.list_indexes()
        index_names = [idx.name for idx in existing_indexes.indexes]
        return name in index_names

    async def list_collections(self) -> list[str]:
        """List all indexes."""
        if not self._client:
            raise ConnectionError("Not connected to Pinecone")

        existing_indexes = self._client.list_indexes()
        return [idx.name for idx in existing_indexes.indexes]

    # -------------------------------------------------------------------------
    # Data Operations
    # -------------------------------------------------------------------------

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        """Insert or update a vector."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        vector_metadata = metadata or {}
        vector_metadata["content"] = content

        self._index.upsert(
            vectors=[(id, embedding, vector_metadata)],
            namespace=namespace or "",
        )
        return id

    async def upsert_batch(
        self,
        items: Sequence[dict[str, Any]],
        namespace: str | None = None,
    ) -> list[str]:
        """Batch upsert vectors."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        vectors = []
        ids = []
        for item in items:
            item_id = item["id"]
            ids.append(item_id)
            metadata = item.get("metadata", {})
            metadata["content"] = item["content"]
            vectors.append((item_id, item["embedding"], metadata))

        # Batch in chunks of 100 (Pinecone limit)
        batch_size = self.config.batch_size or 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=namespace or "")

        return ids

    async def delete(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> int:
        """Delete vectors by ID."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        self._index.delete(ids=list(ids), namespace=namespace or "")
        return len(ids)

    async def delete_by_filter(
        self,
        filters: dict[str, Any],
        namespace: str | None = None,
    ) -> int:
        """Delete vectors by metadata filter."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        # Pinecone requires fetching IDs first, then deleting
        # For now, use the filter parameter in delete (Pinecone serverless feature)
        self._index.delete(filter=filters, namespace=namespace or "")
        return -1  # Count not available

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
        min_score: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        results = self._index.query(
            vector=embedding,
            top_k=limit,
            filter=filters,
            namespace=namespace or "",
            include_metadata=True,
            include_values=False,
        )

        return [
            VectorSearchResult(
                id=match.id,
                content=match.metadata.get("content", "") if match.metadata else "",
                score=match.score,
                metadata={k: v for k, v in (match.metadata or {}).items() if k != "content"},
            )
            for match in results.matches
            if match.score >= min_score
        ]

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[VectorSearchResult]:
        """Hybrid search - Pinecone primarily uses vector search."""
        # Pinecone doesn't have native keyword search
        # Fall back to vector search only
        logger.debug("Pinecone does not support native hybrid search, using vector search")
        return await self.search(embedding, limit, filters, namespace)

    # -------------------------------------------------------------------------
    # Retrieval Operations
    # -------------------------------------------------------------------------

    async def get_by_id(
        self,
        id: str,
        namespace: str | None = None,
    ) -> VectorSearchResult | None:
        """Get a vector by ID."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        results = self._index.fetch(ids=[id], namespace=namespace or "")

        if id not in results.vectors:
            return None

        vector = results.vectors[id]
        return VectorSearchResult(
            id=id,
            content=vector.metadata.get("content", "") if vector.metadata else "",
            score=1.0,
            metadata={k: v for k, v in (vector.metadata or {}).items() if k != "content"},
            embedding=vector.values,
        )

    async def get_by_ids(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> list[VectorSearchResult]:
        """Get multiple vectors by ID."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        results = self._index.fetch(ids=list(ids), namespace=namespace or "")

        return [
            VectorSearchResult(
                id=vid,
                content=vector.metadata.get("content", "") if vector.metadata else "",
                score=1.0,
                metadata={k: v for k, v in (vector.metadata or {}).items() if k != "content"},
                embedding=vector.values,
            )
            for vid, vector in results.vectors.items()
        ]

    async def count(
        self,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> int:
        """Count vectors."""
        if not self._index:
            raise ConnectionError("Not connected to Pinecone")

        stats = self._index.describe_index_stats()

        if namespace:
            ns_stats = stats.namespaces.get(namespace or "", None)
            return ns_stats.vector_count if ns_stats else 0

        return stats.total_vector_count

    # -------------------------------------------------------------------------
    # Health & Diagnostics
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Check connection health."""
        if not self._index:
            return {"status": "disconnected", "error": "Not connected"}

        try:
            stats = self._index.describe_index_stats()
            return {
                "status": "healthy",
                "backend": "pinecone",
                "index": self._index_name,
                "total_vectors": stats.total_vector_count,
                "dimensions": stats.dimension,
                "namespaces": list(stats.namespaces.keys()),
            }
        except Exception as e:
            logger.warning("Pinecone health check failed: %s", e)
            return {"status": "unhealthy", "error": "Health check failed"}
