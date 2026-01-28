"""Knowledge API for the Aragora SDK."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


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
    """Knowledge search result model."""

    id: str
    content: str
    score: float
    source: str | None = None
    metadata: dict[str, Any] | None = None


class KnowledgeStats(BaseModel):
    """Knowledge stats model."""

    total_entries: int = 0
    total_facts: int = 0
    sources: dict[str, int] = {}
    categories: dict[str, int] = {}
    avg_confidence: float = 0.0
    last_updated: str | None = None


class Fact(BaseModel):
    """Knowledge fact model."""

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

    def __init__(self, client) -> None:
        self._client = client
        self._default_workspace_id = "default"

    def _fact_to_entry(self, data: dict[str, Any]) -> KnowledgeEntry:
        """Normalize fact payloads into KnowledgeEntry for backward compatibility."""
        payload = dict(data)
        if "statement" in payload and "content" not in payload:
            payload["content"] = payload.get("statement")
        if "topics" in payload and "tags" not in payload:
            payload["tags"] = payload.get("topics")
        if "source_documents" in payload and "source" not in payload:
            docs = payload.get("source_documents") or []
            payload["source"] = docs[0] if docs else None
        return KnowledgeEntry.model_validate(payload)

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
        workspace_id: str | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Search knowledge chunks via embeddings."""
        params: dict[str, Any] = {"q": query, "limit": limit}
        if min_score is not None:
            params["min_score"] = min_score
        if source_filter:
            params["source"] = source_filter
        if tags:
            params["tags"] = ",".join(tags)
        if workspace_id:
            params["workspace_id"] = workspace_id

        data = await self._client._get("/api/v1/knowledge/search", params=params)
        results = data.get("results", []) if isinstance(data, dict) else data
        return [KnowledgeSearchResult.model_validate(item) for item in results or []]

    async def query(
        self,
        question: str,
        *,
        context: str | None = None,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """Run a natural-language query against the knowledge base."""
        payload: dict[str, Any] = {
            "question": question,
            "include_sources": include_sources,
        }
        if context:
            payload["context"] = context
        return await self._client._post("/api/v1/knowledge/query", payload)

    # ==========================================================================
    # CRUD for Knowledge Entries
    # ==========================================================================

    async def add(
        self,
        content: str,
        *,
        statement: str | None = None,
        workspace_id: str | None = None,
        topics: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        source_documents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        source_type: str | None = None,
        confidence: float | None = None,
    ) -> KnowledgeEntry:
        """Add a new knowledge fact (backwards compatible with KnowledgeEntry)."""
        payload: dict[str, Any] = {
            "statement": statement or content,
            "workspace_id": workspace_id or self._default_workspace_id,
            "confidence": confidence if confidence is not None else 0.5,
        }
        if topics is not None:
            payload["topics"] = topics
        if tags is not None and topics is None:
            payload["topics"] = tags
        if evidence_ids is not None:
            payload["evidence_ids"] = evidence_ids
        if source_documents is not None:
            payload["source_documents"] = source_documents
        elif source is not None:
            payload["source_documents"] = [source]
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        if source_type is not None:
            payload.setdefault("metadata", {})["source_type"] = source_type

        data = await self._client._post("/api/v1/knowledge/facts", payload)
        return self._fact_to_entry(data)

    async def get(self, entry_id: str) -> KnowledgeEntry:
        """Get a knowledge fact by ID."""
        data = await self._client._get(f"/api/v1/knowledge/facts/{entry_id}")
        return self._fact_to_entry(data)

    async def update(
        self,
        entry_id: str,
        *,
        confidence: float | None = None,
        validation_status: str | None = None,
        evidence_ids: list[str] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        superseded_by: str | None = None,
        tags: list[str] | None = None,
        source_type: str | None = None,
    ) -> KnowledgeEntry:
        """Update a knowledge fact."""
        updates: dict[str, Any] = {}
        if confidence is not None:
            updates["confidence"] = confidence
        if validation_status is not None:
            updates["validation_status"] = validation_status
        if evidence_ids is not None:
            updates["evidence_ids"] = evidence_ids
        if topics is not None:
            updates["topics"] = topics
        if tags is not None and topics is None:
            updates["topics"] = tags
        if metadata is not None:
            updates["metadata"] = metadata
        if superseded_by is not None:
            updates["superseded_by"] = superseded_by
        if source_type is not None:
            updates.setdefault("metadata", {})["source_type"] = source_type
        data = await self._client._put(f"/api/v1/knowledge/facts/{entry_id}", updates)
        return self._fact_to_entry(data)

    async def delete(self, entry_id: str) -> dict[str, Any]:
        """Delete a knowledge fact."""
        result = await self._client._delete(f"/api/v1/knowledge/facts/{entry_id}")
        return result if result is not None else {"deleted": True, "fact_id": entry_id}

    # ==========================================================================
    # Fact Operations
    # ==========================================================================

    async def list_facts(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        verified: bool | None = None,
        source: str | None = None,
    ) -> list[Fact]:
        """List facts with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if verified is not None:
            params["verified"] = verified
        if source is not None:
            params["source"] = source
        data = await self._client._get("/api/v1/knowledge/facts", params=params)
        facts = data.get("facts", []) if isinstance(data, dict) else data
        return [Fact.model_validate(item) for item in facts or []]

    async def get_fact(self, fact_id: str) -> Fact:
        """Get a single fact by ID."""
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
        """Add a new fact."""
        payload: dict[str, Any] = {"content": content}
        if source is not None:
            payload["source"] = source
        if confidence is not None:
            payload["confidence"] = confidence
        if metadata is not None:
            payload["metadata"] = metadata
        data = await self._client._post("/api/v1/knowledge/facts", payload)
        return Fact.model_validate(data)

    async def verify_fact(
        self,
        fact_id: str,
        *,
        agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """Verify a fact with agents."""
        payload = {"agents": agents} if agents else {}
        return await self._client._post(
            f"/api/v1/knowledge/facts/{fact_id}/verify",
            payload,
        )

    async def get_contradictions(self, fact_id: str) -> list[Fact]:
        """Get contradictions for a fact."""
        data = await self._client._get(
            f"/api/v1/knowledge/facts/{fact_id}/contradictions"
        )
        contradictions = (
            data.get("contradictions", []) if isinstance(data, dict) else data
        )
        return [Fact.model_validate(item) for item in contradictions or []]

    # ==========================================================================
    # Statistics & Bulk Operations
    # ==========================================================================

    async def get_stats(self) -> KnowledgeStats:
        """Get knowledge base statistics."""
        data = await self._client._get("/api/v1/knowledge/stats")
        return KnowledgeStats.model_validate(data)

    async def bulk_import(
        self,
        entries: list[dict[str, Any]],
        *,
        skip_duplicates: bool = True,
    ) -> dict[str, Any]:
        """Bulk import entries by creating facts sequentially."""
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for entry in entries:
            try:
                statement = entry.get("statement") or entry.get("content") or ""
                response = await self.add(
                    statement,
                    workspace_id=entry.get("workspace_id"),
                    topics=entry.get("topics") or entry.get("tags"),
                    evidence_ids=entry.get("evidence_ids"),
                    source_documents=entry.get("source_documents"),
                    metadata=entry.get("metadata"),
                    source=entry.get("source"),
                    source_type=entry.get("source_type"),
                    confidence=entry.get("confidence"),
                )
                results.append(response.model_dump())
            except Exception as exc:  # pragma: no cover - network failures
                errors.append({"entry": entry, "error": str(exc)})

        return {
            "imported": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
            "skip_duplicates": skip_duplicates,
        }

    # ==========================================================================
    # Compatibility aliases
    # ==========================================================================

    async def create_entry(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Alias for add()."""
        return await self.add(content, **kwargs)

    async def get_entry(self, entry_id: str) -> KnowledgeEntry:
        """Alias for get()."""
        return await self.get(entry_id)

    async def update_entry(self, entry_id: str, **kwargs: Any) -> KnowledgeEntry:
        """Alias for update()."""
        return await self.update(entry_id, **kwargs)

    async def delete_entry(self, entry_id: str) -> dict[str, Any]:
        """Alias for delete()."""
        return await self.delete(entry_id)
