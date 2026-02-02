"""
Facts Namespace API.

Provides REST APIs for fact management:
- CRUD operations for facts
- Relationship management between facts
- Batch operations and search
- Fact validation and merging
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class FactsAPI:
    """Synchronous Facts API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # --- CRUD ---

    def create_fact(
        self,
        content: str,
        source: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new fact."""
        data: dict[str, Any] = {"content": content}
        if source:
            data["source"] = source
        if confidence is not None:
            data["confidence"] = confidence
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        return self._client.request("POST", "/api/v1/facts", json=data)

    def get_fact(self, fact_id: str) -> dict[str, Any]:
        """Get a fact by ID."""
        return self._client.request("GET", f"/api/v1/facts/{fact_id}")

    def update_fact(self, fact_id: str, **updates: Any) -> dict[str, Any]:
        """Update a fact."""
        return self._client.request("PATCH", f"/api/v1/facts/{fact_id}", json=updates)

    def delete_fact(self, fact_id: str) -> dict[str, Any]:
        """Delete a fact."""
        return self._client.request("DELETE", f"/api/v1/facts/{fact_id}")

    def list_facts(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        source: str | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """List facts with optional filters."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if tag:
            params["tag"] = tag
        if source:
            params["source"] = source
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        return self._client.request("GET", "/api/v1/facts", params=params if params else None)

    def search_facts(
        self,
        query: str,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """Search facts by query."""
        params: dict[str, Any] = {"query": query}
        if limit is not None:
            params["limit"] = limit
        if min_score is not None:
            params["min_score"] = min_score
        return self._client.request("GET", "/api/v1/facts/search", params=params)

    def exists(self, fact_id: str) -> dict[str, Any]:
        """Check if a fact exists."""
        return self._client.request("HEAD", f"/api/v1/facts/{fact_id}")

    # --- Relationships ---

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        strength: float | None = None,
    ) -> dict[str, Any]:
        """Create a relationship between facts."""
        data: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
        }
        if strength is not None:
            data["strength"] = strength
        return self._client.request("POST", "/api/v1/facts/relationships", json=data)

    def get_relationship(self, rel_id: str) -> dict[str, Any]:
        """Get a relationship by ID."""
        return self._client.request("GET", f"/api/v1/facts/relationships/{rel_id}")

    def update_relationship(self, rel_id: str, **updates: Any) -> dict[str, Any]:
        """Update a relationship."""
        return self._client.request("PATCH", f"/api/v1/facts/relationships/{rel_id}", json=updates)

    def delete_relationship(self, rel_id: str) -> dict[str, Any]:
        """Delete a relationship."""
        return self._client.request("DELETE", f"/api/v1/facts/relationships/{rel_id}")

    def get_relationships(
        self,
        fact_id: str,
        direction: str | None = None,
        rel_type: str | None = None,
    ) -> dict[str, Any]:
        """Get relationships for a fact."""
        params: dict[str, Any] = {}
        if direction:
            params["direction"] = direction
        if rel_type:
            params["rel_type"] = rel_type
        return self._client.request(
            "GET",
            f"/api/v1/facts/{fact_id}/relationships",
            params=params if params else None,
        )

    def get_related_facts(
        self,
        fact_id: str,
        max_depth: int | None = None,
        min_strength: float | None = None,
    ) -> dict[str, Any]:
        """Get related facts via graph traversal."""
        params: dict[str, Any] = {}
        if max_depth is not None:
            params["max_depth"] = max_depth
        if min_strength is not None:
            params["min_strength"] = min_strength
        return self._client.request(
            "GET",
            f"/api/v1/facts/{fact_id}/related",
            params=params if params else None,
        )

    # --- Batch & Utility ---

    def batch_create(self, facts: list[dict[str, Any]]) -> dict[str, Any]:
        """Create multiple facts in a batch."""
        return self._client.request("POST", "/api/v1/facts/batch", json={"facts": facts})

    def batch_delete(self, ids: list[str]) -> dict[str, Any]:
        """Delete multiple facts in a batch."""
        return self._client.request("POST", "/api/v1/facts/batch/delete", json={"ids": ids})

    def get_stats(self) -> dict[str, Any]:
        """Get fact statistics."""
        return self._client.request("GET", "/api/v1/facts/stats")

    def validate_content(self, content: str) -> dict[str, Any]:
        """Validate fact content."""
        return self._client.request("POST", "/api/v1/facts/validate", json={"content": content})

    def merge_facts(
        self,
        source_id: str,
        target_id: str,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Merge two facts."""
        data: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
        }
        if strategy:
            data["strategy"] = strategy
        return self._client.request("POST", "/api/v1/facts/merge", json=data)


class AsyncFactsAPI:
    """Asynchronous Facts API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def create_fact(
        self,
        content: str,
        source: str | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new fact."""
        data: dict[str, Any] = {"content": content}
        if source:
            data["source"] = source
        if confidence is not None:
            data["confidence"] = confidence
        if tags:
            data["tags"] = tags
        if metadata:
            data["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/facts", json=data)

    async def get_fact(self, fact_id: str) -> dict[str, Any]:
        """Get a fact by ID."""
        return await self._client.request("GET", f"/api/v1/facts/{fact_id}")

    async def update_fact(self, fact_id: str, **updates: Any) -> dict[str, Any]:
        """Update a fact."""
        return await self._client.request("PATCH", f"/api/v1/facts/{fact_id}", json=updates)

    async def delete_fact(self, fact_id: str) -> dict[str, Any]:
        """Delete a fact."""
        return await self._client.request("DELETE", f"/api/v1/facts/{fact_id}")

    async def list_facts(
        self,
        limit: int | None = None,
        offset: int | None = None,
        tag: str | None = None,
        source: str | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """List facts with optional filters."""
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if tag:
            params["tag"] = tag
        if source:
            params["source"] = source
        if min_confidence is not None:
            params["min_confidence"] = min_confidence
        return await self._client.request("GET", "/api/v1/facts", params=params if params else None)

    async def search_facts(
        self,
        query: str,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """Search facts by query."""
        params: dict[str, Any] = {"query": query}
        if limit is not None:
            params["limit"] = limit
        if min_score is not None:
            params["min_score"] = min_score
        return await self._client.request("GET", "/api/v1/facts/search", params=params)

    async def exists(self, fact_id: str) -> dict[str, Any]:
        """Check if a fact exists."""
        return await self._client.request("HEAD", f"/api/v1/facts/{fact_id}")

    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        strength: float | None = None,
    ) -> dict[str, Any]:
        """Create a relationship between facts."""
        data: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "rel_type": rel_type,
        }
        if strength is not None:
            data["strength"] = strength
        return await self._client.request("POST", "/api/v1/facts/relationships", json=data)

    async def get_relationship(self, rel_id: str) -> dict[str, Any]:
        """Get a relationship by ID."""
        return await self._client.request("GET", f"/api/v1/facts/relationships/{rel_id}")

    async def update_relationship(self, rel_id: str, **updates: Any) -> dict[str, Any]:
        """Update a relationship."""
        return await self._client.request(
            "PATCH", f"/api/v1/facts/relationships/{rel_id}", json=updates
        )

    async def delete_relationship(self, rel_id: str) -> dict[str, Any]:
        """Delete a relationship."""
        return await self._client.request("DELETE", f"/api/v1/facts/relationships/{rel_id}")

    async def get_relationships(
        self,
        fact_id: str,
        direction: str | None = None,
        rel_type: str | None = None,
    ) -> dict[str, Any]:
        """Get relationships for a fact."""
        params: dict[str, Any] = {}
        if direction:
            params["direction"] = direction
        if rel_type:
            params["rel_type"] = rel_type
        return await self._client.request(
            "GET",
            f"/api/v1/facts/{fact_id}/relationships",
            params=params if params else None,
        )

    async def get_related_facts(
        self,
        fact_id: str,
        max_depth: int | None = None,
        min_strength: float | None = None,
    ) -> dict[str, Any]:
        """Get related facts via graph traversal."""
        params: dict[str, Any] = {}
        if max_depth is not None:
            params["max_depth"] = max_depth
        if min_strength is not None:
            params["min_strength"] = min_strength
        return await self._client.request(
            "GET",
            f"/api/v1/facts/{fact_id}/related",
            params=params if params else None,
        )

    async def batch_create(self, facts: list[dict[str, Any]]) -> dict[str, Any]:
        """Create multiple facts in a batch."""
        return await self._client.request("POST", "/api/v1/facts/batch", json={"facts": facts})

    async def batch_delete(self, ids: list[str]) -> dict[str, Any]:
        """Delete multiple facts in a batch."""
        return await self._client.request("POST", "/api/v1/facts/batch/delete", json={"ids": ids})

    async def get_stats(self) -> dict[str, Any]:
        """Get fact statistics."""
        return await self._client.request("GET", "/api/v1/facts/stats")

    async def validate_content(self, content: str) -> dict[str, Any]:
        """Validate fact content."""
        return await self._client.request(
            "POST", "/api/v1/facts/validate", json={"content": content}
        )

    async def merge_facts(
        self,
        source_id: str,
        target_id: str,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Merge two facts."""
        data: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
        }
        if strategy:
            data["strategy"] = strategy
        return await self._client.request("POST", "/api/v1/facts/merge", json=data)
