"""
Memory Namespace API

Provides methods for interacting with the multi-tier memory system:
- Fast tier: Immediate context (1 min TTL)
- Medium tier: Session memory (1 hour TTL)
- Slow tier: Cross-session learning (1 day TTL)
- Glacial tier: Long-term patterns (1 week TTL)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

MemoryTier = Literal["fast", "medium", "slow", "glacial"]


class MemoryAPI:
    """
    Synchronous Memory API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> client.memory.store("user_preference", {"theme": "dark"}, tier="medium")
        >>> entry = client.memory.retrieve("user_preference", tier="medium")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def store(
        self,
        key: str,
        value: Any,
        tier: MemoryTier = "medium",
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a value in memory.

        Args:
            key: Unique key for the memory entry
            value: Value to store (must be JSON-serializable)
            tier: Memory tier (fast, medium, slow, glacial)
            ttl_seconds: Custom TTL in seconds (overrides tier default)
            metadata: Optional metadata for the entry

        Returns:
            Stored memory entry with ID and expiration
        """
        data: dict[str, Any] = {"key": key, "value": value, "tier": tier}
        if ttl_seconds is not None:
            data["ttl_seconds"] = ttl_seconds
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/v1/memory/store", json=data)

    def retrieve(
        self,
        key: str,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any] | None:
        """
        Retrieve a value from memory.

        Args:
            key: Key to retrieve
            tier: Specific tier to search (searches all if not specified)

        Returns:
            Memory entry if found, None otherwise
        """
        params: dict[str, Any] = {"key": key}
        if tier:
            params["tier"] = tier

        return self._client.request("GET", "/api/v1/memory/retrieve", params=params)

    def search(
        self,
        query: str,
        tier: MemoryTier | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Semantic search across memory entries.

        Args:
            query: Search query
            tier: Filter by tier
            limit: Maximum results

        Returns:
            Matching memory entries
        """
        params: dict[str, Any] = {"query": query, "limit": limit}
        if tier:
            params["tier"] = tier

        return self._client.request("GET", "/api/v1/memory/search", params=params)

    def delete(self, key: str, tier: MemoryTier | None = None) -> dict[str, Any]:
        """
        Delete a memory entry.

        Args:
            key: Key to delete
            tier: Specific tier (deletes from all if not specified)

        Returns:
            Deletion result
        """
        params: dict[str, Any] = {"key": key}
        if tier:
            params["tier"] = tier

        return self._client.request("DELETE", "/api/v1/memory/delete", params=params)

    def get_tiers(self) -> dict[str, Any]:
        """
        Get status of all memory tiers.

        Returns:
            Tier information including entry counts and TTLs
        """
        return self._client.request("GET", "/api/v1/memory/tiers")

    def flush(self, tier: MemoryTier) -> dict[str, Any]:
        """
        Flush all entries from a specific tier.

        Args:
            tier: Tier to flush

        Returns:
            Flush result with count of deleted entries
        """
        return self._client.request("POST", f"/api/v1/memory/tiers/{tier}/flush")

    def get_stats(self) -> dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Statistics including entry counts per tier, hit rates, etc.
        """
        return self._client.request("GET", "/api/v1/memory/stats")

    def promote(self, key: str, from_tier: MemoryTier, to_tier: MemoryTier) -> dict[str, Any]:
        """
        Promote a memory entry to a longer-lived tier.

        Args:
            key: Key to promote
            from_tier: Current tier
            to_tier: Target tier

        Returns:
            Promoted entry
        """
        return self._client.request(
            "POST",
            "/api/v1/memory/promote",
            json={"key": key, "from_tier": from_tier, "to_tier": to_tier},
        )


class AsyncMemoryAPI:
    """
    Asynchronous Memory API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     await client.memory.store("key", "value", tier="fast")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def store(
        self,
        key: str,
        value: Any,
        tier: MemoryTier = "medium",
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a value in memory."""
        data: dict[str, Any] = {"key": key, "value": value, "tier": tier}
        if ttl_seconds is not None:
            data["ttl_seconds"] = ttl_seconds
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/v1/memory/store", json=data)

    async def retrieve(
        self,
        key: str,
        tier: MemoryTier | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a value from memory."""
        params: dict[str, Any] = {"key": key}
        if tier:
            params["tier"] = tier

        return await self._client.request("GET", "/api/v1/memory/retrieve", params=params)

    async def search(
        self,
        query: str,
        tier: MemoryTier | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Semantic search across memory entries."""
        params: dict[str, Any] = {"query": query, "limit": limit}
        if tier:
            params["tier"] = tier

        return await self._client.request("GET", "/api/v1/memory/search", params=params)

    async def delete(self, key: str, tier: MemoryTier | None = None) -> dict[str, Any]:
        """Delete a memory entry."""
        params: dict[str, Any] = {"key": key}
        if tier:
            params["tier"] = tier

        return await self._client.request("DELETE", "/api/v1/memory/delete", params=params)

    async def get_tiers(self) -> dict[str, Any]:
        """Get status of all memory tiers."""
        return await self._client.request("GET", "/api/v1/memory/tiers")

    async def flush(self, tier: MemoryTier) -> dict[str, Any]:
        """Flush all entries from a specific tier."""
        return await self._client.request("POST", f"/api/v1/memory/tiers/{tier}/flush")

    async def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return await self._client.request("GET", "/api/v1/memory/stats")

    async def promote(self, key: str, from_tier: MemoryTier, to_tier: MemoryTier) -> dict[str, Any]:
        """Promote a memory entry to a longer-lived tier."""
        return await self._client.request(
            "POST",
            "/api/v1/memory/promote",
            json={"key": key, "from_tier": from_tier, "to_tier": to_tier},
        )
