"""
Milvus vector store adapter.

Provides integration with Milvus vector database for the Knowledge Mound.

Requirements:
    pip install pymilvus>=2.3
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

# Check for pymilvus library
try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    connections = None
    Collection = None
    logger.debug("pymilvus not available - install with: pip install pymilvus")


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus vector store adapter.

    Provides open-source vector storage with:
    - Distributed architecture
    - GPU acceleration
    - Multiple index types (IVF, HNSW, etc.)
    - Attribute filtering
    - Partition-based multi-tenancy

    Configuration:
        MILVUS_HOST: Milvus server host
        MILVUS_PORT: Milvus server port
        MILVUS_USER: Username (optional)
        MILVUS_PASSWORD: Password (optional)

    Usage:
        config = VectorStoreConfig(
            backend=VectorBackend.MILVUS,
            url="http://localhost:19530",
            collection_name="knowledge_mound",
        )
        store = MilvusVectorStore(config)
        await store.connect()
    """

    def __init__(self, config: VectorStoreConfig):
        """Initialize Milvus store."""
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus client not installed. Install with: pip install pymilvus")

        config.backend = VectorBackend.MILVUS
        super().__init__(config)

        self._alias = "default"
        self._collection: Collection | None = None
        self._collection_name = config.collection_name.replace("-", "_")  # Milvus naming

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to Milvus."""
        if self._connected:
            return

        try:
            # Parse URL or use host/port
            if self.config.url:
                # Extract host:port from URL
                url = self.config.url.replace("http://", "").replace("https://", "")
                parts = url.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 19530
            else:
                host = os.environ.get("MILVUS_HOST", "localhost")
                port = int(os.environ.get("MILVUS_PORT", "19530"))

            user = self.config.extra.get("user", os.environ.get("MILVUS_USER", ""))
            password = self.config.extra.get("password", os.environ.get("MILVUS_PASSWORD", ""))

            # Connect to Milvus
            connections.connect(
                alias=self._alias,
                host=host,
                port=port,
                user=user if user else None,
                password=password if password else None,
            )

            # Create collection if it doesn't exist
            if not utility.has_collection(self._collection_name, using=self._alias):
                await self.create_collection(self._collection_name)

            self._collection = Collection(
                name=self._collection_name,
                using=self._alias,
            )

            # Load collection into memory for search
            self._collection.load()

            self._connected = True
            logger.info(f"Connected to Milvus at {host}:{port}")

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Milvus: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to Milvus."""
        if self._collection:
            try:
                self._collection.release()
            except Exception as e:
                logger.warning(f"Error releasing Milvus collection: {e}")

        try:
            connections.disconnect(self._alias)
        except Exception as e:
            logger.warning(f"Error disconnecting from Milvus: {e}")
        finally:
            self._collection = None
            self._connected = False

    def _get_metric_type(self, metric: str) -> str:
        """Map generic metric name to Milvus metric type."""
        metric_map = {
            "cosine": "COSINE",
            "euclidean": "L2",
            "dot_product": "IP",
        }
        return metric_map.get(metric, "COSINE")

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    async def create_collection(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Create a new collection."""
        if utility.has_collection(name, using=self._alias):
            return

        # Define schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=256,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.embedding_dimensions,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="namespace",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="metadata_json",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
        ]

        collection_schema = CollectionSchema(
            fields=fields,
            description=f"Knowledge Mound collection: {name}",
        )

        collection = Collection(
            name=name,
            schema=collection_schema,
            using=self._alias,
        )

        # Create index for vector field
        index_params = {
            "metric_type": self._get_metric_type(self.config.distance_metric),
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        logger.info(f"Created Milvus collection: {name}")

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            if utility.has_collection(name, using=self._alias):
                utility.drop_collection(name, using=self._alias)
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete Milvus collection {name}: {e}")
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        return utility.has_collection(name, using=self._alias)

    async def list_collections(self) -> list[str]:
        """List all collections."""
        return utility.list_collections(using=self._alias)

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
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        import json

        # Delete existing if present (Milvus doesn't have native upsert)
        self._collection.delete(expr=f'id == "{id}"')

        # Insert new
        entities = [
            [id],  # id
            [embedding],  # embedding
            [content],  # content
            [namespace or ""],  # namespace
            [json.dumps(metadata or {})],  # metadata_json
        ]

        self._collection.insert(entities)
        self._collection.flush()
        return id

    async def upsert_batch(
        self,
        items: Sequence[dict[str, Any]],
        namespace: str | None = None,
    ) -> list[str]:
        """Batch upsert vectors."""
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        import json

        ids = [item["id"] for item in items]

        # Delete existing (batch)
        if ids:
            id_list = ", ".join(f'"{i}"' for i in ids)
            self._collection.delete(expr=f"id in [{id_list}]")

        # Prepare batch data
        entities = [
            ids,  # id
            [item["embedding"] for item in items],  # embedding
            [item["content"] for item in items],  # content
            [namespace or "" for _ in items],  # namespace
            [json.dumps(item.get("metadata", {})) for item in items],  # metadata_json
        ]

        # Insert in batches
        batch_size = self.config.batch_size or 1000
        for i in range(0, len(ids), batch_size):
            batch_entities = [e[i : i + batch_size] for e in entities]
            self._collection.insert(batch_entities)

        self._collection.flush()
        return ids

    async def delete(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> int:
        """Delete vectors by ID."""
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        id_list = ", ".join(f'"{i}"' for i in ids)
        expr = f"id in [{id_list}]"

        if namespace:
            expr += f' and namespace == "{namespace}"'

        result = self._collection.delete(expr=expr)
        self._collection.flush()
        return result.delete_count if hasattr(result, "delete_count") else len(ids)

    async def delete_by_filter(
        self,
        filters: dict[str, Any],
        namespace: str | None = None,
    ) -> int:
        """Delete vectors by filter."""
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        # Build expression from filters
        expressions = []
        if namespace:
            expressions.append(f'namespace == "{namespace}"')

        for key, value in filters.items():
            if isinstance(value, str):
                expressions.append(f'{key} == "{value}"')
            else:
                expressions.append(f"{key} == {value}")

        if not expressions:
            return 0

        expr = " and ".join(expressions)
        result = self._collection.delete(expr=expr)
        self._collection.flush()
        return result.delete_count if hasattr(result, "delete_count") else 0

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
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        import json

        # Build filter expression
        expr = None
        if namespace or filters:
            expressions = []
            if namespace:
                expressions.append(f'namespace == "{namespace}"')
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        expressions.append(f'{key} == "{value}"')
                    else:
                        expressions.append(f"{key} == {value}")
            expr = " and ".join(expressions)

        search_params = {
            "metric_type": self._get_metric_type(self.config.distance_metric),
            "params": {"ef": 64},
        }

        results = self._collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["id", "content", "namespace", "metadata_json"],
        )

        search_results = []
        for hits in results:
            for hit in hits:
                # Convert distance to score (higher is better)
                score = 1.0 / (1.0 + hit.distance) if hit.distance >= 0 else 0.0

                if score >= min_score:
                    metadata = {}
                    if hit.entity.get("metadata_json"):
                        try:
                            metadata = json.loads(hit.entity.get("metadata_json", "{}"))
                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse JSON data: %s", e)

                    search_results.append(
                        VectorSearchResult(
                            id=hit.entity.get("id", str(hit.id)),
                            content=hit.entity.get("content", ""),
                            score=score,
                            metadata=metadata,
                        )
                    )

        return search_results

    async def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        limit: int = 10,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[VectorSearchResult]:
        """Hybrid search - Milvus primarily uses vector search."""
        # Milvus 2.x doesn't have native full-text search
        # Fall back to vector search
        logger.debug("Milvus does not support native hybrid search, using vector search")
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
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        import json

        expr = f'id == "{id}"'
        if namespace:
            expr += f' and namespace == "{namespace}"'

        results = self._collection.query(
            expr=expr,
            output_fields=["id", "content", "embedding", "metadata_json"],
        )

        if not results:
            return None

        hit = results[0]
        metadata = {}
        if hit.get("metadata_json"):
            try:
                metadata = json.loads(hit.get("metadata_json", "{}"))
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON data: %s", e)

        return VectorSearchResult(
            id=hit.get("id", id),
            content=hit.get("content", ""),
            score=1.0,
            metadata=metadata,
            embedding=hit.get("embedding"),
        )

    async def get_by_ids(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> list[VectorSearchResult]:
        """Get multiple vectors by ID."""
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        import json

        id_list = ", ".join(f'"{i}"' for i in ids)
        expr = f"id in [{id_list}]"

        if namespace:
            expr += f' and namespace == "{namespace}"'

        results = self._collection.query(
            expr=expr,
            output_fields=["id", "content", "embedding", "metadata_json"],
        )

        search_results = []
        for hit in results:
            metadata = {}
            if hit.get("metadata_json"):
                try:
                    metadata = json.loads(hit.get("metadata_json", "{}"))
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse JSON data: %s", e)

            search_results.append(
                VectorSearchResult(
                    id=hit.get("id", ""),
                    content=hit.get("content", ""),
                    score=1.0,
                    metadata=metadata,
                    embedding=hit.get("embedding"),
                )
            )

        return search_results

    async def count(
        self,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> int:
        """Count vectors."""
        if not self._collection:
            raise ConnectionError("Not connected to Milvus")

        self._collection.flush()

        if not filters and not namespace:
            return self._collection.num_entities

        # Build filter and count via query
        expressions = []
        if namespace:
            expressions.append(f'namespace == "{namespace}"')
        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    expressions.append(f'{key} == "{value}"')
                else:
                    expressions.append(f"{key} == {value}")

        expr = " and ".join(expressions) if expressions else ""
        results = self._collection.query(expr=expr, output_fields=["id"])
        return len(results)

    # -------------------------------------------------------------------------
    # Health & Diagnostics
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Check connection health."""
        if not self._collection:
            return {"status": "disconnected", "error": "Not connected"}

        try:
            self._collection.flush()
            stats = utility.get_collection_stats(self._collection_name, using=self._alias)

            return {
                "status": "healthy",
                "backend": "milvus",
                "collection": self._collection_name,
                "total_vectors": self._collection.num_entities,
                "row_count": stats.get("row_count", 0),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
