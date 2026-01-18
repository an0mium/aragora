"""
Chroma vector store adapter.

Provides integration with ChromaDB for the Knowledge Mound.
Chroma is lightweight and can run embedded, making it ideal
for development and smaller deployments.

Requirements:
    pip install chromadb>=0.4
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

# Check for chromadb library
try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.debug("chromadb not available - install with: pip install chromadb")


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB vector store adapter.

    Provides lightweight vector storage with:
    - Embedded mode (no external server needed)
    - Persistent storage to disk
    - Simple metadata filtering
    - Built-in embedding functions (optional)

    Usage:
        config = VectorStoreConfig(
            backend=VectorBackend.CHROMA,
            url="./chroma_data",  # Persistence path
            collection_name="knowledge_mound",
        )
        store = ChromaVectorStore(config)
        await store.connect()
    """

    def __init__(self, config: VectorStoreConfig):
        """Initialize Chroma store."""
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        config.backend = VectorBackend.CHROMA
        super().__init__(config)

        self._client: Optional[chromadb.ClientAPI] = None
        self._collections: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to Chroma."""
        if self._connected:
            return

        try:
            persist_path = self.config.url or "./chroma_data"

            # Check if we should use HTTP client or persistent client
            if persist_path.startswith("http"):
                # HTTP client for remote server
                self._client = chromadb.HttpClient(
                    host=persist_path.replace("http://", "").replace("https://", "").split(":")[0],
                    port=int(persist_path.split(":")[-1]) if ":" in persist_path.split("/")[-1] else 8000,
                )
            else:
                # Persistent client with local storage
                self._client = chromadb.PersistentClient(
                    path=persist_path,
                    settings=Settings(anonymized_telemetry=False),
                )

            self._connected = True

            # Ensure default collection exists
            if not await self.collection_exists(self.config.collection_name):
                await self.create_collection(self.config.collection_name)

            logger.info(f"Connected to Chroma at {persist_path}")

        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to Chroma: {e}") from e

    async def disconnect(self) -> None:
        """Close connection to Chroma."""
        self._client = None
        self._collections.clear()
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
            raise ConnectionError("Not connected to Chroma")

        # Map distance metric
        distance_map = {
            "cosine": "cosine",
            "euclidean": "l2",
            "dot_product": "ip",
        }
        distance = distance_map.get(self.config.distance_metric, "cosine")

        # Get or create collection
        self._collections[name] = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": distance},
        )

        logger.info(f"Created Chroma collection: {name}")

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if not self._client:
            raise ConnectionError("Not connected to Chroma")

        try:
            self._client.delete_collection(name)
            self._collections.pop(name, None)
            return True
        except Exception:
            return False

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        if not self._client:
            raise ConnectionError("Not connected to Chroma")

        try:
            self._client.get_collection(name)
            return True
        except Exception:
            return False

    async def list_collections(self) -> list[str]:
        """List all collections."""
        if not self._client:
            raise ConnectionError("Not connected to Chroma")

        collections = self._client.list_collections()
        return [c.name for c in collections]

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

        # Chroma stores metadata separately
        meta = {
            "namespace": namespace or "",
            **(metadata or {}),
        }

        # Ensure all metadata values are valid types for Chroma
        meta = self._sanitize_metadata(meta)

        collection.upsert(
            ids=[id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
        )

        return id

    async def upsert_batch(
        self,
        items: Sequence[dict[str, Any]],
        namespace: Optional[str] = None,
    ) -> list[str]:
        """Batch upsert multiple vectors."""
        collection = self._get_collection()

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for item in items:
            item_id = item.get("id") or str(uuid_lib.uuid4())
            ids.append(item_id)
            embeddings.append(item["embedding"])
            documents.append(item["content"])

            meta = {
                "namespace": namespace or "",
                **(item.get("metadata") or {}),
            }
            metadatas.append(self._sanitize_metadata(meta))

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return ids

    async def delete(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete vectors by ID."""
        collection = self._get_collection()

        collection.delete(ids=list(ids))
        return len(ids)

    async def delete_by_filter(
        self,
        filters: dict[str, Any],
        namespace: Optional[str] = None,
    ) -> int:
        """Delete vectors matching filter criteria."""
        collection = self._get_collection()

        where = self._build_filter(filters, namespace)

        # Get IDs matching filter
        results = collection.get(where=where, include=[])
        ids = results.get("ids", [])

        if ids:
            collection.delete(ids=ids)

        return len(ids)

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

        where = self._build_filter(filters, namespace)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        search_results = []

        # Chroma returns nested lists
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0] if results.get("embeddings") else [None] * len(ids)

        for i, id in enumerate(ids):
            # Convert distance to similarity (Chroma returns distance, not similarity)
            # For cosine: similarity = 1 - distance
            distance = distances[i] if distances else 0
            score = max(0, 1 - distance)

            if score >= min_score:
                meta = metadatas[i] if metadatas else {}
                search_results.append(
                    VectorSearchResult(
                        id=id,
                        content=documents[i] if documents else "",
                        score=score,
                        metadata={
                            k: v for k, v in meta.items() if k != "namespace"
                        },
                        embedding=embeddings[i] if embeddings and embeddings[i] else None,
                    )
                )

        return search_results

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

        Note: Chroma doesn't have built-in BM25, so we approximate
        by boosting results that contain query terms.
        """
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
        collection = self._get_collection()

        try:
            results = collection.get(
                ids=[id],
                include=["documents", "metadatas", "embeddings"],
            )

            if results["ids"]:
                meta = results["metadatas"][0] if results["metadatas"] else {}
                return VectorSearchResult(
                    id=results["ids"][0],
                    content=results["documents"][0] if results["documents"] else "",
                    score=1.0,
                    metadata={k: v for k, v in meta.items() if k != "namespace"},
                    embedding=results["embeddings"][0] if results.get("embeddings") else None,
                )
        except Exception as e:
            logger.debug(f"Error retrieving vector by ID: {e}")

        return None

    async def get_by_ids(
        self,
        ids: Sequence[str],
        namespace: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """Get multiple vectors by ID."""
        collection = self._get_collection()

        results = collection.get(
            ids=list(ids),
            include=["documents", "metadatas", "embeddings"],
        )

        search_results = []
        for i, id in enumerate(results.get("ids", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            search_results.append(
                VectorSearchResult(
                    id=id,
                    content=results["documents"][i] if results.get("documents") else "",
                    score=1.0,
                    metadata={k: v for k, v in meta.items() if k != "namespace"},
                    embedding=results["embeddings"][i] if results.get("embeddings") else None,
                )
            )

        return search_results

    async def count(
        self,
        filters: Optional[dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Count vectors matching optional filters."""
        collection = self._get_collection()

        if not filters and not namespace:
            return collection.count()

        where = self._build_filter(filters, namespace)
        results = collection.get(where=where, include=[])
        return len(results.get("ids", []))

    # -------------------------------------------------------------------------
    # Health & Diagnostics
    # -------------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Check health and return diagnostics."""
        if not self._client:
            return {"status": "disconnected", "backend": "chroma"}

        try:
            collections = self._client.list_collections()
            return {
                "status": "healthy",
                "backend": "chroma",
                "collections": len(collections),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "chroma",
                "error": str(e),
            }

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_collection(self) -> Any:
        """Get collection reference."""
        if not self._client:
            raise ConnectionError("Not connected to Chroma")

        name = self.config.collection_name
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name)
        return self._collections[name]

    def _build_filter(
        self,
        filters: Optional[dict[str, Any]],
        namespace: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build Chroma filter from dict."""
        conditions = []

        if namespace:
            conditions.append({"namespace": {"$eq": namespace}})

        if filters:
            for key, value in filters.items():
                conditions.append({key: {"$eq": value}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def _sanitize_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Ensure metadata values are valid for Chroma."""
        sanitized = {}
        for key, value in meta.items():
            # Chroma only accepts str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            else:
                # Convert to string
                sanitized[key] = str(value)
        return sanitized
