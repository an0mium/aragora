"""
Vector Index Namespace API

Provides methods for vector indexing and semantic search:
- Vector embedding generation
- Semantic similarity search
- Index management and operations
- Batch embedding processing
- Document management within indexes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

_List = list  # Preserve builtin list for type annotations


class VectorIndexAPI:
    """
    Synchronous Vector Index API.

    Provides vector embedding generation, semantic search, and index
    management capabilities.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> result = client.vector_index.embed(texts=["hello world", "machine learning"])
        >>> search = client.vector_index.search(
        ...     query="AI safety",
        ...     documents=["neural networks", "safety constraints", "ethics"],
        ... )
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Embedding Operations
    # =========================================================================

    def embed(
        self,
        text: str | None = None,
        texts: _List[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate vector embeddings for text.

        Provide either a single text or a list of texts (max 100).

        Args:
            text: Single text to embed
            texts: List of texts to embed (max 100)
            model: Optional model name to use for embedding

        Returns:
            Dict with embeddings array and dimension
        """
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if texts is not None:
            data["texts"] = texts
        if model is not None:
            data["model"] = model
        return self._client.request("POST", "/api/v1/ml/embed", json=data)

    def embed_batch(
        self,
        texts: _List[str],
        model: str | None = None,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Generate embeddings for a large batch of texts.

        Args:
            texts: List of texts to embed
            model: Optional model name to use
            batch_size: Number of texts per batch (max 100)

        Returns:
            Dict with embeddings, dimension, and batch_count
        """
        data: dict[str, Any] = {
            "texts": texts,
            "batch_size": min(batch_size, 100),
        }
        if model is not None:
            data["model"] = model
        return self._client.request("POST", "/api/v1/index/embed-batch", json=data)

    # =========================================================================
    # Semantic Search
    # =========================================================================

    def search(
        self,
        query: str,
        documents: _List[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Perform semantic similarity search.

        Args:
            query: Search query text
            documents: List of documents to search (max 1000)
            top_k: Number of results to return (default 5)
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            Dict with results containing text, score, and index
        """
        return self._client.request(
            "POST",
            "/api/v1/ml/search",
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k,
                "threshold": threshold,
            },
        )

    def search_index(
        self,
        query: str,
        index_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search a named vector index.

        Args:
            query: Search query text
            index_name: Name of the index to search
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Dict with results containing text, score, and metadata
        """
        data: dict[str, Any] = {
            "query": query,
            "index_name": index_name,
            "top_k": top_k,
        }
        if filters is not None:
            data["filters"] = filters
        return self._client.request("POST", "/api/v1/index/search", json=data)

    # =========================================================================
    # Index Management
    # =========================================================================

    def list_indexes(self) -> dict[str, Any]:
        """List all available vector indexes."""
        return self._client.request("GET", "/api/v1/index")

    def get_index(self, index_name: str) -> dict[str, Any]:
        """Get details of a specific index."""
        return self._client.request("GET", f"/api/v1/index/{index_name}")

    def create_index(
        self,
        name: str,
        dimension: int | None = None,
        metric: str = "cosine",
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a new vector index.

        Args:
            name: Name for the new index
            dimension: Vector dimension (auto-detected if not specified)
            metric: Distance metric (cosine, euclidean, dot_product)
            description: Optional description

        Returns:
            Created index with name, status, created_at
        """
        data: dict[str, Any] = {"name": name, "metric": metric}
        if dimension is not None:
            data["dimension"] = dimension
        if description is not None:
            data["description"] = description
        return self._client.request("POST", "/api/v1/index", json=data)

    def delete_index(self, index_name: str) -> dict[str, Any]:
        """Delete a vector index."""
        return self._client.request("DELETE", f"/api/v1/index/{index_name}")

    def get_index_stats(self, index_name: str) -> dict[str, Any]:
        """Get statistics for a vector index."""
        return self._client.request("GET", f"/api/v1/index/{index_name}/stats")

    # =========================================================================
    # Document Operations
    # =========================================================================

    def add_documents(self, index_name: str, documents: _List[dict[str, Any]]) -> dict[str, Any]:
        """Add documents to an index."""
        return self._client.request(
            "POST", f"/api/v1/index/{index_name}/documents", json={"documents": documents}
        )

    def update_document(
        self, index_name: str, document_id: str, text: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Update a document in an index."""
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if metadata is not None:
            data["metadata"] = metadata
        return self._client.request("PUT", f"/api/v1/index/{index_name}/documents/{document_id}", json=data)

    def delete_documents(self, index_name: str, document_ids: _List[str]) -> dict[str, Any]:
        """Delete documents from an index."""
        return self._client.request(
            "DELETE", f"/api/v1/index/{index_name}/documents", json={"document_ids": document_ids}
        )

    # =========================================================================
    # Index Operations
    # =========================================================================

    def rebuild_index(self, index_name: str) -> dict[str, Any]:
        """Rebuild an index from scratch."""
        return self._client.request("POST", f"/api/v1/index/{index_name}/rebuild")

    def optimize_index(self, index_name: str) -> dict[str, Any]:
        """Optimize an index for better search performance."""
        return self._client.request("POST", f"/api/v1/index/{index_name}/optimize")


class AsyncVectorIndexAPI:
    """
    Asynchronous Vector Index API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.vector_index.embed(texts=["hello", "world"])
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def embed(
        self, text: str | None = None, texts: _List[str] | None = None, model: str | None = None
    ) -> dict[str, Any]:
        """Generate vector embeddings for text."""
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if texts is not None:
            data["texts"] = texts
        if model is not None:
            data["model"] = model
        return await self._client.request("POST", "/api/v1/ml/embed", json=data)

    async def embed_batch(
        self, texts: _List[str], model: str | None = None, batch_size: int = 100
    ) -> dict[str, Any]:
        """Generate embeddings for a large batch of texts."""
        data: dict[str, Any] = {"texts": texts, "batch_size": min(batch_size, 100)}
        if model is not None:
            data["model"] = model
        return await self._client.request("POST", "/api/v1/index/embed-batch", json=data)

    async def search(
        self, query: str, documents: _List[str], top_k: int = 5, threshold: float = 0.0
    ) -> dict[str, Any]:
        """Perform semantic similarity search."""
        return await self._client.request(
            "POST", "/api/v1/ml/search",
            json={"query": query, "documents": documents, "top_k": top_k, "threshold": threshold},
        )

    async def search_index(
        self, query: str, index_name: str, top_k: int = 10, filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Search a named vector index."""
        data: dict[str, Any] = {"query": query, "index_name": index_name, "top_k": top_k}
        if filters is not None:
            data["filters"] = filters
        return await self._client.request("POST", "/api/v1/index/search", json=data)

    async def list_indexes(self) -> dict[str, Any]:
        """List all available vector indexes."""
        return await self._client.request("GET", "/api/v1/index")

    async def get_index(self, index_name: str) -> dict[str, Any]:
        """Get details of a specific index."""
        return await self._client.request("GET", f"/api/v1/index/{index_name}")

    async def create_index(
        self, name: str, dimension: int | None = None, metric: str = "cosine", description: str | None = None
    ) -> dict[str, Any]:
        """Create a new vector index."""
        data: dict[str, Any] = {"name": name, "metric": metric}
        if dimension is not None:
            data["dimension"] = dimension
        if description is not None:
            data["description"] = description
        return await self._client.request("POST", "/api/v1/index", json=data)

    async def delete_index(self, index_name: str) -> dict[str, Any]:
        """Delete a vector index."""
        return await self._client.request("DELETE", f"/api/v1/index/{index_name}")

    async def get_index_stats(self, index_name: str) -> dict[str, Any]:
        """Get statistics for a vector index."""
        return await self._client.request("GET", f"/api/v1/index/{index_name}/stats")

    async def add_documents(self, index_name: str, documents: _List[dict[str, Any]]) -> dict[str, Any]:
        """Add documents to an index."""
        return await self._client.request(
            "POST", f"/api/v1/index/{index_name}/documents", json={"documents": documents}
        )

    async def update_document(
        self, index_name: str, document_id: str, text: str | None = None, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Update a document in an index."""
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if metadata is not None:
            data["metadata"] = metadata
        return await self._client.request(
            "PUT", f"/api/v1/index/{index_name}/documents/{document_id}", json=data
        )

    async def delete_documents(self, index_name: str, document_ids: _List[str]) -> dict[str, Any]:
        """Delete documents from an index."""
        return await self._client.request(
            "DELETE", f"/api/v1/index/{index_name}/documents", json={"document_ids": document_ids}
        )

    async def rebuild_index(self, index_name: str) -> dict[str, Any]:
        """Rebuild an index from scratch."""
        return await self._client.request("POST", f"/api/v1/index/{index_name}/rebuild")

    async def optimize_index(self, index_name: str) -> dict[str, Any]:
        """Optimize an index for better search performance."""
        return await self._client.request("POST", f"/api/v1/index/{index_name}/optimize")
