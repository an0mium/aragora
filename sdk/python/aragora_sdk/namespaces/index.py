"""
Index Namespace API.

Provides vector index and semantic search operations.

Features:
- Vector embedding generation
- Semantic similarity search
- Index management and operations
- Batch embedding processing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

IndexStatus = Literal["ready", "building", "updating", "error"]


class IndexAPI:
    """
    Synchronous Index API.

    Provides methods for vector indexing and semantic search:
    - Generate embeddings for text
    - Perform semantic similarity search
    - Manage vector indexes
    - Batch processing for large datasets

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> # Generate embeddings
        >>> result = client.index.embed(texts=["hello world", "machine learning"])
        >>> # Semantic search
        >>> results = client.index.search(
        ...     query="AI safety",
        ...     documents=["neural networks", "safety constraints", "ethics"],
        ... )
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Embedding Operations
    # =========================================================================

    def embed(
        self,
        text: str | None = None,
        texts: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate vector embeddings for text.

        Provide either a single text or a list of texts (max 100).

        Args:
            text: Single text to embed.
            texts: List of texts to embed (max 100).
            model: Optional model name to use for embedding.

        Returns:
            Dict with embeddings array and dimension.
        """
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if texts is not None:
            data["texts"] = texts
        if model is not None:
            data["model"] = model

        return self._client._request("POST", "/api/v1/ml/embed", json=data)

    def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """
        Generate embeddings for a large batch of texts.

        Automatically handles batching for large datasets.

        Args:
            texts: List of texts to embed.
            model: Optional model name to use.
            batch_size: Number of texts per batch (max 100).

        Returns:
            Dict with all embeddings and metadata.
        """
        data: dict[str, Any] = {
            "texts": texts,
            "batch_size": min(batch_size, 100),
        }
        if model is not None:
            data["model"] = model

        return self._client._request("POST", "/api/v1/index/embed-batch", json=data)

    # =========================================================================
    # Semantic Search
    # =========================================================================

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """
        Perform semantic similarity search.

        Args:
            query: Search query text.
            documents: List of documents to search (max 1000).
            top_k: Number of results to return (default 5).
            threshold: Minimum similarity score (0.0-1.0, default 0.0).

        Returns:
            Dict with results containing text, score, and index.
        """
        return self._client._request(
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
        """
        Search a named vector index.

        Args:
            query: Search query text.
            index_name: Name of the index to search.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            Dict with matching documents and scores.
        """
        data: dict[str, Any] = {
            "query": query,
            "index_name": index_name,
            "top_k": top_k,
        }
        if filters is not None:
            data["filters"] = filters

        return self._client._request("POST", "/api/v1/index/search", json=data)

    # =========================================================================
    # Index Management
    # =========================================================================

    def list_indexes(self) -> dict[str, Any]:
        """
        List all available vector indexes.

        Returns:
            Dict with list of indexes and their metadata.
        """
        return self._client._request("GET", "/api/v1/index")

    def get_index(self, index_name: str) -> dict[str, Any]:
        """
        Get details of a specific index.

        Args:
            index_name: Name of the index.

        Returns:
            Dict with index details including status, document count, and dimension.
        """
        return self._client._request("GET", f"/api/v1/index/{index_name}")

    def create_index(
        self,
        name: str,
        dimension: int | None = None,
        metric: str = "cosine",
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new vector index.

        Args:
            name: Name for the new index.
            dimension: Vector dimension (auto-detected if not specified).
            metric: Distance metric (cosine, euclidean, dot_product).
            description: Optional description.

        Returns:
            Dict with created index details.
        """
        data: dict[str, Any] = {
            "name": name,
            "metric": metric,
        }
        if dimension is not None:
            data["dimension"] = dimension
        if description is not None:
            data["description"] = description

        return self._client._request("POST", "/api/v1/index", json=data)

    def delete_index(self, index_name: str) -> dict[str, Any]:
        """
        Delete a vector index.

        Args:
            index_name: Name of the index to delete.

        Returns:
            Dict confirming deletion.
        """
        return self._client._request("DELETE", f"/api/v1/index/{index_name}")

    def get_index_stats(self, index_name: str) -> dict[str, Any]:
        """
        Get statistics for a vector index.

        Args:
            index_name: Name of the index.

        Returns:
            Dict with statistics including document count, size, and performance metrics.
        """
        return self._client._request("GET", f"/api/v1/index/{index_name}/stats")

    # =========================================================================
    # Document Operations
    # =========================================================================

    def add_documents(
        self,
        index_name: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Add documents to an index.

        Args:
            index_name: Name of the target index.
            documents: List of documents with 'text' and optional 'metadata'.

        Returns:
            Dict with number of documents added.
        """
        return self._client._request(
            "POST",
            f"/api/v1/index/{index_name}/documents",
            json={"documents": documents},
        )

    def update_document(
        self,
        index_name: str,
        document_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update a document in an index.

        Args:
            index_name: Name of the index.
            document_id: ID of the document to update.
            text: New text content (re-embeds if changed).
            metadata: Updated metadata.

        Returns:
            Dict confirming the update.
        """
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if metadata is not None:
            data["metadata"] = metadata

        return self._client._request(
            "PUT",
            f"/api/v1/index/{index_name}/documents/{document_id}",
            json=data,
        )

    def delete_documents(
        self,
        index_name: str,
        document_ids: list[str],
    ) -> dict[str, Any]:
        """
        Delete documents from an index.

        Args:
            index_name: Name of the index.
            document_ids: List of document IDs to delete.

        Returns:
            Dict with number of documents deleted.
        """
        return self._client._request(
            "DELETE",
            f"/api/v1/index/{index_name}/documents",
            json={"document_ids": document_ids},
        )

    # =========================================================================
    # Index Operations
    # =========================================================================

    def rebuild_index(self, index_name: str) -> dict[str, Any]:
        """
        Rebuild an index from scratch.

        Args:
            index_name: Name of the index to rebuild.

        Returns:
            Dict with rebuild job status.
        """
        return self._client._request("POST", f"/api/v1/index/{index_name}/rebuild")

    def optimize_index(self, index_name: str) -> dict[str, Any]:
        """
        Optimize an index for better search performance.

        Args:
            index_name: Name of the index to optimize.

        Returns:
            Dict with optimization status.
        """
        return self._client._request("POST", f"/api/v1/index/{index_name}/optimize")


class AsyncIndexAPI:
    """
    Asynchronous Index API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     embeddings = await client.index.embed(texts=["hello", "world"])
        ...     print(f"Dimension: {embeddings['dimension']}")
    """

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Embedding Operations
    # =========================================================================

    async def embed(
        self,
        text: str | None = None,
        texts: list[str] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Generate vector embeddings for text."""
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if texts is not None:
            data["texts"] = texts
        if model is not None:
            data["model"] = model

        return await self._client._request("POST", "/api/v1/ml/embed", json=data)

    async def embed_batch(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """Generate embeddings for a large batch of texts."""
        data: dict[str, Any] = {
            "texts": texts,
            "batch_size": min(batch_size, 100),
        }
        if model is not None:
            data["model"] = model

        return await self._client._request("POST", "/api/v1/index/embed-batch", json=data)

    # =========================================================================
    # Semantic Search
    # =========================================================================

    async def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> dict[str, Any]:
        """Perform semantic similarity search."""
        return await self._client._request(
            "POST",
            "/api/v1/ml/search",
            json={
                "query": query,
                "documents": documents,
                "top_k": top_k,
                "threshold": threshold,
            },
        )

    async def search_index(
        self,
        query: str,
        index_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search a named vector index."""
        data: dict[str, Any] = {
            "query": query,
            "index_name": index_name,
            "top_k": top_k,
        }
        if filters is not None:
            data["filters"] = filters

        return await self._client._request("POST", "/api/v1/index/search", json=data)

    # =========================================================================
    # Index Management
    # =========================================================================

    async def list_indexes(self) -> dict[str, Any]:
        """List all available vector indexes."""
        return await self._client._request("GET", "/api/v1/index")

    async def get_index(self, index_name: str) -> dict[str, Any]:
        """Get details of a specific index."""
        return await self._client._request("GET", f"/api/v1/index/{index_name}")

    async def create_index(
        self,
        name: str,
        dimension: int | None = None,
        metric: str = "cosine",
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a new vector index."""
        data: dict[str, Any] = {
            "name": name,
            "metric": metric,
        }
        if dimension is not None:
            data["dimension"] = dimension
        if description is not None:
            data["description"] = description

        return await self._client._request("POST", "/api/v1/index", json=data)

    async def delete_index(self, index_name: str) -> dict[str, Any]:
        """Delete a vector index."""
        return await self._client._request("DELETE", f"/api/v1/index/{index_name}")

    async def get_index_stats(self, index_name: str) -> dict[str, Any]:
        """Get statistics for a vector index."""
        return await self._client._request("GET", f"/api/v1/index/{index_name}/stats")

    # =========================================================================
    # Document Operations
    # =========================================================================

    async def add_documents(
        self,
        index_name: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Add documents to an index."""
        return await self._client._request(
            "POST",
            f"/api/v1/index/{index_name}/documents",
            json={"documents": documents},
        )

    async def update_document(
        self,
        index_name: str,
        document_id: str,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a document in an index."""
        data: dict[str, Any] = {}
        if text is not None:
            data["text"] = text
        if metadata is not None:
            data["metadata"] = metadata

        return await self._client._request(
            "PUT",
            f"/api/v1/index/{index_name}/documents/{document_id}",
            json=data,
        )

    async def delete_documents(
        self,
        index_name: str,
        document_ids: list[str],
    ) -> dict[str, Any]:
        """Delete documents from an index."""
        return await self._client._request(
            "DELETE",
            f"/api/v1/index/{index_name}/documents",
            json={"document_ids": document_ids},
        )

    # =========================================================================
    # Index Operations
    # =========================================================================

    async def rebuild_index(self, index_name: str) -> dict[str, Any]:
        """Rebuild an index from scratch."""
        return await self._client._request("POST", f"/api/v1/index/{index_name}/rebuild")

    async def optimize_index(self, index_name: str) -> dict[str, Any]:
        """Optimize an index for better search performance."""
        return await self._client._request("POST", f"/api/v1/index/{index_name}/optimize")
