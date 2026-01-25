"""Knowledge API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


class KnowledgeEntry(BaseModel):
    """Knowledge entry model."""

    id: str | None = None
    content: str
    source: str | None = None
    source_type: str | None = None
    metadata: dict[str, Any] | None = None
    confidence: float | None = None
    created_at: str | None = None
    updated_at: str | None = None
    tags: list[str] | None = None


class KnowledgeSearchResult(BaseModel):
    """Knowledge search result."""

    id: str
    content: str
    score: float
    source: str | None = None
    metadata: dict[str, Any] | None = None


class KnowledgeStats(BaseModel):
    """Knowledge base statistics."""

    total_entries: int = 0
    total_facts: int = 0
    sources: dict[str, int] | None = None
    categories: dict[str, int] | None = None
    avg_confidence: float | None = None
    last_updated: str | None = None


class Fact(BaseModel):
    """Fact model."""

    id: str
    content: str
    source: str | None = None
    confidence: float | None = None
    verified: bool = False
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None


class KnowledgeAPI:
    """API for knowledge base operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # ==========================================================================
    # Search and Query
    # ==========================================================================

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        min_score: float | None = None,
        source_filter: str | None = None,
        tags: list[str] | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Search the knowledge base.

        Args:
            query: Search query string
            limit: Maximum number of results
            min_score: Minimum similarity score threshold
            source_filter: Filter by source
            tags: Filter by tags

        Returns:
            List of search results with scores
        """
        params: dict[str, Any] = {"query": query, "limit": limit}
        if min_score is not None:
            params["min_score"] = min_score
        if source_filter:
            params["source"] = source_filter
        if tags:
            params["tags"] = ",".join(tags)

        data = await self._client._get("/api/v1/knowledge/search", params=params)
        return [
            KnowledgeSearchResult.model_validate(r) for r in data.get("results", [])
        ]

    async def query(
        self,
        question: str,
        *,
        context: str | None = None,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """Query the knowledge base with natural language.

        Args:
            question: Natural language question
            context: Additional context for the query
            include_sources: Whether to include source citations

        Returns:
            Query response with answer and sources
        """
        body: dict[str, Any] = {
            "question": question,
            "include_sources": include_sources,
        }
        if context:
            body["context"] = context

        return await self._client._post("/api/v1/knowledge/query", body)

    # ==========================================================================
    # CRUD Operations
    # ==========================================================================

    async def add(
        self,
        content: str,
        *,
        source: str | None = None,
        source_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Add an entry to the knowledge base.

        Args:
            content: The knowledge content
            source: Source of the knowledge
            source_type: Type of source (document, url, manual, etc.)
            metadata: Additional metadata
            tags: Tags for categorization
            confidence: Confidence score

        Returns:
            Created entry info with id and timestamp
        """
        entry = KnowledgeEntry(
            content=content,
            source=source,
            source_type=source_type,
            metadata=metadata,
            tags=tags,
            confidence=confidence,
        )
        return await self._client._post(
            "/api/v1/knowledge", entry.model_dump(exclude_none=True)
        )

    async def get(self, entry_id: str) -> KnowledgeEntry:
        """Get a knowledge entry by ID.

        Args:
            entry_id: Knowledge entry ID

        Returns:
            Knowledge entry
        """
        data = await self._client._get(f"/api/v1/knowledge/{entry_id}")
        return KnowledgeEntry.model_validate(data)

    async def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        confidence: float | None = None,
    ) -> KnowledgeEntry:
        """Update a knowledge entry.

        Args:
            entry_id: Knowledge entry ID
            content: Updated content
            metadata: Updated metadata
            tags: Updated tags
            confidence: Updated confidence

        Returns:
            Updated knowledge entry
        """
        updates: dict[str, Any] = {}
        if content is not None:
            updates["content"] = content
        if metadata is not None:
            updates["metadata"] = metadata
        if tags is not None:
            updates["tags"] = tags
        if confidence is not None:
            updates["confidence"] = confidence

        data = await self._client._put(f"/api/v1/knowledge/{entry_id}", updates)
        return KnowledgeEntry.model_validate(data)

    async def delete(self, entry_id: str) -> dict[str, bool]:
        """Delete a knowledge entry.

        Args:
            entry_id: Knowledge entry ID

        Returns:
            Deletion confirmation
        """
        await self._client._delete(f"/api/v1/knowledge/{entry_id}")
        return {"deleted": True}

    # ==========================================================================
    # Facts Operations
    # ==========================================================================

    async def list_facts(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        verified: bool | None = None,
        source: str | None = None,
    ) -> list[Fact]:
        """List facts from the knowledge base.

        Args:
            limit: Maximum number of facts to return
            offset: Pagination offset
            verified: Filter by verification status
            source: Filter by source

        Returns:
            List of facts
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verified is not None:
            params["verified"] = verified
        if source:
            params["source"] = source

        data = await self._client._get("/api/v1/knowledge/facts", params=params)
        return [Fact.model_validate(f) for f in data.get("facts", [])]

    async def get_fact(self, fact_id: str) -> Fact:
        """Get a specific fact.

        Args:
            fact_id: Fact ID

        Returns:
            Fact details
        """
        data = await self._client._get(f"/api/v1/knowledge/facts/{fact_id}")
        return Fact.model_validate(data)

    async def add_fact(
        self,
        content: str,
        *,
        source: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Add a fact to the knowledge base.

        Args:
            content: Fact content
            source: Source of the fact
            confidence: Confidence score
            metadata: Additional metadata

        Returns:
            Created fact
        """
        body: dict[str, Any] = {"content": content}
        if source:
            body["source"] = source
        if confidence is not None:
            body["confidence"] = confidence
        if metadata:
            body["metadata"] = metadata

        data = await self._client._post("/api/v1/knowledge/facts", body)
        return Fact.model_validate(data)

    async def verify_fact(
        self,
        fact_id: str,
        *,
        agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """Verify a fact using agents.

        Args:
            fact_id: Fact ID to verify
            agents: Optional list of agents to use for verification

        Returns:
            Verification result
        """
        body: dict[str, Any] = {}
        if agents:
            body["agents"] = agents

        return await self._client._post(
            f"/api/v1/knowledge/facts/{fact_id}/verify", body
        )

    async def get_contradictions(self, fact_id: str) -> list[Fact]:
        """Get facts that contradict the given fact.

        Args:
            fact_id: Fact ID

        Returns:
            List of contradicting facts
        """
        data = await self._client._get(
            f"/api/v1/knowledge/facts/{fact_id}/contradictions"
        )
        return [Fact.model_validate(f) for f in data.get("contradictions", [])]

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> KnowledgeStats:
        """Get knowledge base statistics.

        Returns:
            Knowledge base statistics
        """
        data = await self._client._get("/api/v1/knowledge/stats")
        return KnowledgeStats.model_validate(data)

    # ==========================================================================
    # Bulk Operations
    # ==========================================================================

    async def bulk_import(
        self,
        entries: list[dict[str, Any]],
        *,
        skip_duplicates: bool = True,
    ) -> dict[str, Any]:
        """Bulk import knowledge entries.

        Args:
            entries: List of knowledge entries to import
            skip_duplicates: Whether to skip duplicate entries

        Returns:
            Import results with counts
        """
        body = {
            "entries": entries,
            "skip_duplicates": skip_duplicates,
        }
        return await self._client._post("/api/v1/knowledge/bulk-import", body)
