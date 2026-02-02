"""
RLM (Recursive Language Models) Namespace API

Provides API endpoints for RLM compression and query operations:
- Content compression with hierarchical abstraction
- Query operations on compressed contexts
- Context storage and retrieval
- Streaming with multiple modes

RLM enables programmatic interaction with long contexts by treating
them as external environment variables rather than direct prompt input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

# Type aliases
RLMStrategy = Literal["peek", "grep", "partition_map", "summarize", "hierarchical", "auto"]
SourceType = Literal["text", "code", "debate"]
StreamMode = Literal["top_down", "bottom_up", "targeted", "progressive"]


class RLMAPI:
    """
    Synchronous RLM API.

    Provides methods for RLM compression and query operations:
    - Content compression with hierarchical abstraction
    - Query operations on compressed contexts
    - Context storage and retrieval
    - Streaming with multiple modes

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> stats = client.rlm.get_stats()
        >>> result = client.rlm.compress(content="Long document...", source_type="text")
        >>> answer = client.rlm.query(result["context_id"], "What is the main topic?")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Statistics & Configuration
    # ===========================================================================

    def get_stats(self) -> dict[str, Any]:
        """
        Get RLM compression statistics.

        Returns:
            Dict with cache, contexts, system, and timestamp fields
        """
        return self._client.request("GET", "/api/v1/rlm/stats")

    def get_strategies(self) -> dict[str, Any]:
        """
        Get available decomposition strategies.

        Returns:
            Dict with strategies (name → info), default strategy, and documentation URL
        """
        return self._client.request("GET", "/api/v1/rlm/strategies")

    def get_stream_modes(self) -> dict[str, Any]:
        """
        Get available streaming modes.

        Returns:
            Dict with modes array containing mode, description, and use_case
        """
        return self._client.request("GET", "/api/v1/rlm/stream/modes")

    # ===========================================================================
    # Compression
    # ===========================================================================

    def compress(
        self,
        content: str,
        source_type: SourceType = "text",
        levels: int = 4,
    ) -> dict[str, Any]:
        """
        Compress content and get a context ID for later querying.

        Args:
            content: The content to compress
            source_type: Type of content ('text', 'code', 'debate')
            levels: Number of abstraction levels (default: 4)

        Returns:
            Dict with context_id, compression_result, and created_at
        """
        data: dict[str, Any] = {
            "content": content,
            "source_type": source_type,
            "levels": levels,
        }
        return self._client.request("POST", "/api/v1/rlm/compress", json=data)

    # ===========================================================================
    # Query
    # ===========================================================================

    def query(
        self,
        context_id: str,
        query: str,
        strategy: RLMStrategy = "auto",
        refine: bool = False,
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """
        Query a compressed context.

        Args:
            context_id: ID of the compressed context
            query: The question to answer
            strategy: Decomposition strategy to use
            refine: Whether to refine the answer iteratively
            max_iterations: Maximum refinement iterations

        Returns:
            Dict with answer, metadata (context_id, strategy, refined, etc.), and timestamp
        """
        data: dict[str, Any] = {
            "context_id": context_id,
            "query": query,
            "strategy": strategy,
            "refine": refine,
            "max_iterations": max_iterations,
        }
        return self._client.request("POST", "/api/v1/rlm/query", json=data)

    # ===========================================================================
    # Context Management
    # ===========================================================================

    def list_contexts(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        List stored compressed contexts.

        Args:
            limit: Maximum contexts to return
            offset: Pagination offset

        Returns:
            Dict with contexts array, total, limit, and offset
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request("GET", "/api/v1/rlm/contexts", params=params or None)

    def get_context(
        self,
        context_id: str,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """
        Get details of a specific context.

        Args:
            context_id: Context ID
            include_content: Include summary preview content

        Returns:
            Dict with id, source_type, original_tokens, compressed_tokens,
            compression_ratio, levels, and optionally summary_preview
        """
        params: dict[str, Any] = {}
        if include_content:
            params["include_content"] = True
        return self._client.request(
            "GET", f"/api/v1/rlm/context/{context_id}", params=params or None
        )

    def delete_context(self, context_id: str) -> dict[str, Any]:
        """
        Delete a compressed context.

        Args:
            context_id: Context ID to delete

        Returns:
            Dict with success, context_id, and message
        """
        return self._client.request("DELETE", f"/api/v1/rlm/context/{context_id}")

    # ===========================================================================
    # Streaming
    # ===========================================================================

    def stream(
        self,
        context_id: str,
        mode: StreamMode = "top_down",
        query: str | None = None,
        level: str | None = None,
        chunk_size: int = 500,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """
        Stream context with configurable modes.

        Args:
            context_id: Context ID to stream
            mode: Streaming mode ('top_down', 'bottom_up', 'targeted', 'progressive')
            query: Query string for targeted mode
            level: Specific abstraction level to stream
            chunk_size: Size of chunks (default: 500)
            include_metadata: Include metadata in chunks (default: True)

        Returns:
            Dict with context_id, mode, query, chunks array, total_chunks, and timestamp
        """
        data: dict[str, Any] = {
            "context_id": context_id,
            "mode": mode,
            "chunk_size": chunk_size,
            "include_metadata": include_metadata,
        }
        if query is not None:
            data["query"] = query
        if level is not None:
            data["level"] = level
        return self._client.request("POST", "/api/v1/rlm/stream", json=data)

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    def compress_and_query(
        self,
        content: str,
        query: str,
        source_type: SourceType = "text",
        strategy: RLMStrategy = "auto",
    ) -> dict[str, Any]:
        """
        Convenience method to compress content and query in one call.

        Args:
            content: The content to compress
            query: The question to answer
            source_type: Type of content
            strategy: Decomposition strategy

        Returns:
            Dict with query_result (answer, metadata) and context_id
        """
        # Compress first
        compress_result = self.compress(content, source_type=source_type)
        context_id = compress_result["context_id"]

        # Then query
        query_result = self.query(context_id, query, strategy=strategy)

        return {
            "query_result": query_result,
            "context_id": context_id,
        }


class AsyncRLMAPI:
    """
    Asynchronous RLM API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     result = await client.rlm.compress(content="Long document...")
        ...     answer = await client.rlm.query(result["context_id"], "What is the main topic?")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Statistics & Configuration
    # ===========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get RLM compression statistics."""
        return await self._client.request("GET", "/api/v1/rlm/stats")

    async def get_strategies(self) -> dict[str, Any]:
        """Get available decomposition strategies."""
        return await self._client.request("GET", "/api/v1/rlm/strategies")

    async def get_stream_modes(self) -> dict[str, Any]:
        """Get available streaming modes."""
        return await self._client.request("GET", "/api/v1/rlm/stream/modes")

    # ===========================================================================
    # Compression
    # ===========================================================================

    async def compress(
        self,
        content: str,
        source_type: SourceType = "text",
        levels: int = 4,
    ) -> dict[str, Any]:
        """Compress content and get a context ID."""
        data: dict[str, Any] = {
            "content": content,
            "source_type": source_type,
            "levels": levels,
        }
        return await self._client.request("POST", "/api/v1/rlm/compress", json=data)

    # ===========================================================================
    # Query
    # ===========================================================================

    async def query(
        self,
        context_id: str,
        query: str,
        strategy: RLMStrategy = "auto",
        refine: bool = False,
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """Query a compressed context."""
        data: dict[str, Any] = {
            "context_id": context_id,
            "query": query,
            "strategy": strategy,
            "refine": refine,
            "max_iterations": max_iterations,
        }
        return await self._client.request("POST", "/api/v1/rlm/query", json=data)

    # ===========================================================================
    # Context Management
    # ===========================================================================

    async def list_contexts(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List stored compressed contexts."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request("GET", "/api/v1/rlm/contexts", params=params or None)

    async def get_context(
        self,
        context_id: str,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Get details of a specific context."""
        params: dict[str, Any] = {}
        if include_content:
            params["include_content"] = True
        return await self._client.request(
            "GET", f"/api/v1/rlm/context/{context_id}", params=params or None
        )

    async def delete_context(self, context_id: str) -> dict[str, Any]:
        """Delete a compressed context."""
        return await self._client.request("DELETE", f"/api/v1/rlm/context/{context_id}")

    # ===========================================================================
    # Streaming
    # ===========================================================================

    async def stream(
        self,
        context_id: str,
        mode: StreamMode = "top_down",
        query: str | None = None,
        level: str | None = None,
        chunk_size: int = 500,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Stream context with configurable modes."""
        data: dict[str, Any] = {
            "context_id": context_id,
            "mode": mode,
            "chunk_size": chunk_size,
            "include_metadata": include_metadata,
        }
        if query is not None:
            data["query"] = query
        if level is not None:
            data["level"] = level
        return await self._client.request("POST", "/api/v1/rlm/stream", json=data)

    # ===========================================================================
    # Convenience Methods
    # ===========================================================================

    async def compress_and_query(
        self,
        content: str,
        query: str,
        source_type: SourceType = "text",
        strategy: RLMStrategy = "auto",
    ) -> dict[str, Any]:
        """Convenience method to compress content and query in one call."""
        # Compress first
        compress_result = await self.compress(content, source_type=source_type)
        context_id = compress_result["context_id"]

        # Then query
        query_result = await self.query(context_id, query, strategy=strategy)

        return {
            "query_result": query_result,
            "context_id": context_id,
        }
