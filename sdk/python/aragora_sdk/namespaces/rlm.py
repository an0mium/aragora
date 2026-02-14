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
