"""Knowledge API for the Aragora SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Fact(BaseModel):
    """Knowledge fact model."""

    id: str
    statement: str
    confidence: float | None = None
    validation_status: str | None = None
    evidence_ids: list[str] | None = None
    consensus_proof_id: str | None = None
    source_documents: list[str] | None = None
    workspace_id: str | None = None
    topics: list[str] | None = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    superseded_by: str | None = None


class KnowledgeAPI:
    """API for knowledge base operations."""

    def __init__(self, client) -> None:
        self._client = client

    # ==========================================================================
    # Search and Query
    # ==========================================================================

    async def search(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search knowledge chunks via embeddings."""
        params: dict[str, Any] = {"q": query, "limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client._get("/api/v1/knowledge/search", params=params)

    async def query(self, prompt: str) -> dict[str, Any]:
        """Run a natural-language query against the knowledge base."""
        return await self._client._post("/api/v1/knowledge/query", {"prompt": prompt})

    # ==========================================================================
    # Fact Operations
    # ==========================================================================

    async def list_facts(
        self,
        *,
        workspace_id: str | None = None,
        topic: str | None = None,
        min_confidence: float = 0.0,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List facts with filtering."""
        params: dict[str, Any] = {
            "min_confidence": min_confidence,
            "include_superseded": include_superseded,
            "limit": limit,
            "offset": offset,
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        if topic:
            params["topic"] = topic
        if status:
            params["status"] = status
        return await self._client._get("/api/v1/knowledge/facts", params=params)

    async def get_fact(self, fact_id: str) -> Fact:
        """Get a single fact by ID."""
        data = await self._client._get(f"/api/v1/knowledge/facts/{fact_id}")
        return Fact.model_validate(data)

    async def create_fact(
        self,
        statement: str,
        *,
        workspace_id: str = "default",
        confidence: float = 0.5,
        topics: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        source_documents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Create a new fact."""
        payload: dict[str, Any] = {
            "statement": statement,
            "workspace_id": workspace_id,
            "confidence": confidence,
        }
        if topics is not None:
            payload["topics"] = topics
        if evidence_ids is not None:
            payload["evidence_ids"] = evidence_ids
        if source_documents is not None:
            payload["source_documents"] = source_documents
        if metadata is not None:
            payload["metadata"] = metadata
        data = await self._client._post("/api/v1/knowledge/facts", payload)
        return Fact.model_validate(data)

    async def update_fact(
        self,
        fact_id: str,
        *,
        confidence: float | None = None,
        validation_status: str | None = None,
        evidence_ids: list[str] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        superseded_by: str | None = None,
    ) -> Fact:
        """Update an existing fact."""
        updates: dict[str, Any] = {}
        if confidence is not None:
            updates["confidence"] = confidence
        if validation_status is not None:
            updates["validation_status"] = validation_status
        if evidence_ids is not None:
            updates["evidence_ids"] = evidence_ids
        if topics is not None:
            updates["topics"] = topics
        if metadata is not None:
            updates["metadata"] = metadata
        if superseded_by is not None:
            updates["superseded_by"] = superseded_by
        data = await self._client._put(f"/api/v1/knowledge/facts/{fact_id}", updates)
        return Fact.model_validate(data)

    async def delete_fact(self, fact_id: str) -> dict[str, Any]:
        """Delete a fact by ID."""
        return await self._client._delete(f"/api/v1/knowledge/facts/{fact_id}")

    async def verify_fact(self, fact_id: str) -> dict[str, Any]:
        """Verify a fact with agents."""
        return await self._client._post(f"/api/v1/knowledge/facts/{fact_id}/verify", {})

    async def list_contradictions(self, fact_id: str) -> dict[str, Any]:
        """Get contradictions for a fact."""
        return await self._client._get(
            f"/api/v1/knowledge/facts/{fact_id}/contradictions"
        )

    async def list_relations(
        self,
        fact_id: str,
        *,
        relation_type: str | None = None,
        as_source: bool = True,
        as_target: bool = True,
    ) -> dict[str, Any]:
        """Get relations for a fact."""
        params: dict[str, Any] = {"as_source": as_source, "as_target": as_target}
        if relation_type:
            params["type"] = relation_type
        return await self._client._get(
            f"/api/v1/knowledge/facts/{fact_id}/relations", params=params
        )

    async def add_relation(
        self,
        fact_id: str,
        *,
        target_fact_id: str,
        relation_type: str,
    ) -> dict[str, Any]:
        """Add a relation from a fact to another fact."""
        payload = {"target_fact_id": target_fact_id, "relation_type": relation_type}
        return await self._client._post(
            f"/api/v1/knowledge/facts/{fact_id}/relations", payload
        )

    async def add_relation_between_facts(
        self,
        *,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: str,
        confidence: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a relation between two facts."""
        payload: dict[str, Any] = {
            "source_fact_id": source_fact_id,
            "target_fact_id": target_fact_id,
            "relation_type": relation_type,
            "confidence": confidence,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client._post("/api/v1/knowledge/facts/relations", payload)

    async def get_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get knowledge base statistics."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client._get("/api/v1/knowledge/stats", params=params)

    async def get_mound_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get Knowledge Mound statistics (if enabled)."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client._get("/api/v1/knowledge/mound/stats", params=params)
