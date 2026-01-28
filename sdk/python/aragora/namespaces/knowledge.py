"""
Knowledge Namespace API

Provides methods for interacting with the Knowledge Mound:
- Semantic search across organizational knowledge
- Knowledge item retrieval and validation
- Federation status and domain management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class KnowledgeAPI:
    """
    Synchronous Knowledge API.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> results = client.knowledge.search("authentication best practices")
        >>> for item in results["items"]:
        ...     print(item["title"], item["confidence"])
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    def search(
        self,
        query: str,
        domain: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        Semantic search across knowledge items.

        Args:
            query: Search query text
            domain: Filter by knowledge domain (optional)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Search results with items and metadata
        """
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        if domain:
            params["domain"] = domain
        if min_confidence is not None:
            params["min_confidence"] = min_confidence

        return self._client.request("GET", "/api/v1/knowledge/search", params=params)

    def get(self, item_id: str) -> dict[str, Any]:
        """
        Get a knowledge item by ID.

        Args:
            item_id: The knowledge item ID

        Returns:
            Knowledge item details
        """
        return self._client.request("GET", f"/api/v1/knowledge/items/{item_id}")

    def list(
        self,
        domain: str | None = None,
        source_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List knowledge items with filtering.

        Args:
            domain: Filter by domain
            source_type: Filter by source type (debate, document, etc.)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of knowledge items with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if domain:
            params["domain"] = domain
        if source_type:
            params["source_type"] = source_type

        return self._client.request("GET", "/api/v1/knowledge/items", params=params)

    def validate(self, item_id: str) -> dict[str, Any]:
        """
        Validate a knowledge item's integrity and freshness.

        Args:
            item_id: The knowledge item ID

        Returns:
            Validation result with confidence and status
        """
        return self._client.request("POST", f"/api/v1/knowledge/items/{item_id}/validate")

    def get_provenance(self, item_id: str) -> dict[str, Any]:
        """
        Get provenance chain for a knowledge item.

        Args:
            item_id: The knowledge item ID

        Returns:
            Provenance information (source debates, evidence, etc.)
        """
        return self._client.request("GET", f"/api/v1/knowledge/items/{item_id}/provenance")

    def get_related(
        self,
        item_id: str,
        relationship_type: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Get related knowledge items.

        Args:
            item_id: The knowledge item ID
            relationship_type: Filter by relationship (supports, contradicts, etc.)
            limit: Maximum related items

        Returns:
            List of related items with relationship info
        """
        params: dict[str, Any] = {"limit": limit}
        if relationship_type:
            params["relationship_type"] = relationship_type

        return self._client.request(
            "GET", f"/api/v1/knowledge/items/{item_id}/related", params=params
        )

    def list_domains(self) -> dict[str, Any]:
        """
        List all knowledge domains.

        Returns:
            List of domains with item counts
        """
        return self._client.request("GET", "/api/v1/knowledge/domains")

    def get_domain(self, domain: str) -> dict[str, Any]:
        """
        Get details for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Domain details with statistics
        """
        return self._client.request("GET", f"/api/v1/knowledge/domains/{domain}")

    def get_federation_status(self) -> dict[str, Any]:
        """
        Get knowledge federation status.

        Returns:
            Federation status including sync state and connected sources
        """
        return self._client.request("GET", "/api/v1/knowledge/federation/status")

    def get_stats(self) -> dict[str, Any]:
        """
        Get knowledge mound statistics.

        Returns:
            Statistics including item counts, domain distribution, etc.
        """
        return self._client.request("GET", "/api/v1/knowledge/stats")


class AsyncKnowledgeAPI:
    """
    Asynchronous Knowledge API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     results = await client.knowledge.search("authentication")
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def search(
        self,
        query: str,
        domain: str | None = None,
        min_confidence: float | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Semantic search across knowledge items."""
        params: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
        if domain:
            params["domain"] = domain
        if min_confidence is not None:
            params["min_confidence"] = min_confidence

        return await self._client.request("GET", "/api/v1/knowledge/search", params=params)

    async def get(self, item_id: str) -> dict[str, Any]:
        """Get a knowledge item by ID."""
        return await self._client.request("GET", f"/api/v1/knowledge/items/{item_id}")

    async def list(
        self,
        domain: str | None = None,
        source_type: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List knowledge items with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if domain:
            params["domain"] = domain
        if source_type:
            params["source_type"] = source_type

        return await self._client.request("GET", "/api/v1/knowledge/items", params=params)

    async def validate(self, item_id: str) -> dict[str, Any]:
        """Validate a knowledge item's integrity and freshness."""
        return await self._client.request("POST", f"/api/v1/knowledge/items/{item_id}/validate")

    async def get_provenance(self, item_id: str) -> dict[str, Any]:
        """Get provenance chain for a knowledge item."""
        return await self._client.request("GET", f"/api/v1/knowledge/items/{item_id}/provenance")

    async def get_related(
        self,
        item_id: str,
        relationship_type: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get related knowledge items."""
        params: dict[str, Any] = {"limit": limit}
        if relationship_type:
            params["relationship_type"] = relationship_type

        return await self._client.request(
            "GET", f"/api/v1/knowledge/items/{item_id}/related", params=params
        )

    async def list_domains(self) -> dict[str, Any]:
        """List all knowledge domains."""
        return await self._client.request("GET", "/api/v1/knowledge/domains")

    async def get_domain(self, domain: str) -> dict[str, Any]:
        """Get details for a specific domain."""
        return await self._client.request("GET", f"/api/v1/knowledge/domains/{domain}")

    async def get_federation_status(self) -> dict[str, Any]:
        """Get knowledge federation status."""
        return await self._client.request("GET", "/api/v1/knowledge/federation/status")

    async def get_stats(self) -> dict[str, Any]:
        """Get knowledge mound statistics."""
        return await self._client.request("GET", "/api/v1/knowledge/stats")
