"""
Knowledge Namespace API

Provides methods for interacting with the Knowledge Base (facts + search):
- Fact CRUD operations
- Fact relations and contradictions
- Natural language queries and semantic search
- Knowledge base statistics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class KnowledgeAPI:
    """Synchronous Knowledge Base API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def search(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search knowledge chunks via embeddings."""
        params: dict[str, Any] = {"q": query, "limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/knowledge/search", params=params)

    def query(self, prompt: str) -> dict[str, Any]:
        """Run a natural-language query against the knowledge base."""
        return self._client.request("POST", "/api/v1/knowledge/query", json={"prompt": prompt})

    def list_facts(
        self,
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
        return self._client.request("GET", "/api/v1/knowledge/facts", params=params)

    def list_all_facts(
        self,
        workspace_id: str | None = None,
        topic: str | None = None,
        min_confidence: float = 0.0,
        status: str | None = None,
        include_superseded: bool = False,
        page_size: int = 50,
    ):
        """
        Iterate through all facts with automatic pagination.

        Args:
            workspace_id: Filter by workspace
            topic: Filter by topic
            min_confidence: Minimum confidence threshold
            status: Filter by status
            include_superseded: Include superseded facts
            page_size: Number of facts per page (default 50)

        Returns:
            SyncPaginator yielding fact dictionaries

        Example::

            for fact in client.knowledge.list_all_facts(topic="security"):
                print(fact["statement"])
        """
        from ..pagination import SyncPaginator

        params: dict[str, Any] = {
            "min_confidence": min_confidence,
            "include_superseded": include_superseded,
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        if topic:
            params["topic"] = topic
        if status:
            params["status"] = status

        return SyncPaginator(self._client, "/api/v1/knowledge/facts", params, page_size)

    def get_fact(self, fact_id: str) -> dict[str, Any]:
        """Get a single fact by ID."""
        return self._client.request("GET", f"/api/v1/knowledge/facts/{fact_id}")

    def create_fact(
        self,
        statement: str,
        workspace_id: str = "default",
        confidence: float = 0.5,
        topics: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        source_documents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        return self._client.request("POST", "/api/v1/knowledge/facts", json=payload)

    def update_fact(
        self,
        fact_id: str,
        confidence: float | None = None,
        validation_status: str | None = None,
        evidence_ids: list[str] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing fact."""
        payload: dict[str, Any] = {}
        if confidence is not None:
            payload["confidence"] = confidence
        if validation_status is not None:
            payload["validation_status"] = validation_status
        if evidence_ids is not None:
            payload["evidence_ids"] = evidence_ids
        if topics is not None:
            payload["topics"] = topics
        if metadata is not None:
            payload["metadata"] = metadata
        if superseded_by is not None:
            payload["superseded_by"] = superseded_by
        return self._client.request("PUT", f"/api/v1/knowledge/facts/{fact_id}", json=payload)

    def delete_fact(self, fact_id: str) -> dict[str, Any]:
        """Delete a fact by ID."""
        return self._client.request("DELETE", f"/api/v1/knowledge/facts/{fact_id}")

    def verify_fact(self, fact_id: str) -> dict[str, Any]:
        """Verify a fact with agents."""
        return self._client.request("POST", f"/api/v1/knowledge/facts/{fact_id}/verify")

    def list_contradictions(self, fact_id: str) -> dict[str, Any]:
        """Get contradictions for a fact."""
        return self._client.request("GET", f"/api/v1/knowledge/facts/{fact_id}/contradictions")

    def list_relations(
        self,
        fact_id: str,
        relation_type: str | None = None,
        as_source: bool = True,
        as_target: bool = True,
    ) -> dict[str, Any]:
        """Get relations for a fact."""
        params: dict[str, Any] = {"as_source": as_source, "as_target": as_target}
        if relation_type:
            params["type"] = relation_type
        return self._client.request(
            "GET", f"/api/v1/knowledge/facts/{fact_id}/relations", params=params
        )

    def add_relation(
        self,
        fact_id: str,
        target_fact_id: str,
        relation_type: str,
    ) -> dict[str, Any]:
        """Add a relation from a fact to another fact."""
        payload = {"target_fact_id": target_fact_id, "relation_type": relation_type}
        return self._client.request(
            "POST", f"/api/v1/knowledge/facts/{fact_id}/relations", json=payload
        )

    def add_relation_between_facts(
        self,
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
        return self._client.request("POST", "/api/v1/knowledge/facts/relations", json=payload)

    def get_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get knowledge base statistics."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/knowledge/stats", params=params)

    def get_mound_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get Knowledge Mound statistics (if enabled)."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/v1/knowledge/mound/stats", params=params)

    # ========== Knowledge Mound Node Operations ==========

    def mound_query(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Semantic query against Knowledge Mound."""
        payload: dict[str, Any] = {"query": query, "limit": limit}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/knowledge/mound/query", json=payload)

    def add_node(
        self,
        content: str,
        node_type: str = "fact",
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a knowledge node to the mound."""
        payload: dict[str, Any] = {"content": content, "node_type": node_type}
        if metadata:
            payload["metadata"] = metadata
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/knowledge/mound/nodes", json=payload)

    def get_node(self, node_id: str) -> dict[str, Any]:
        """Get a specific knowledge node."""
        return self._client.request("GET", f"/api/knowledge/mound/nodes/{node_id}")

    def list_nodes(
        self,
        workspace_id: str | None = None,
        node_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List knowledge nodes with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if node_type:
            params["node_type"] = node_type
        return self._client.request("GET", "/api/knowledge/mound/nodes", params=params)

    def get_node_relationships(self, node_id: str) -> dict[str, Any]:
        """Get relationships for a knowledge node."""
        return self._client.request("GET", f"/api/knowledge/mound/nodes/{node_id}/relationships")

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a relationship between nodes."""
        payload: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
        }
        if metadata:
            payload["metadata"] = metadata
        return self._client.request("POST", "/api/knowledge/mound/relationships", json=payload)

    # ========== Graph Operations ==========

    def get_graph(
        self,
        node_id: str,
        depth: int = 2,
        include_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get graph traversal from a node."""
        params: dict[str, Any] = {"depth": depth}
        if include_types:
            params["include_types"] = ",".join(include_types)
        return self._client.request("GET", f"/api/knowledge/mound/graph/{node_id}", params=params)

    def get_lineage(self, node_id: str) -> dict[str, Any]:
        """Get lineage (provenance chain) for a node."""
        return self._client.request("GET", f"/api/knowledge/mound/graph/{node_id}/lineage")

    def get_related(
        self,
        node_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get semantically related nodes."""
        params: dict[str, Any] = {"limit": limit}
        return self._client.request(
            "GET", f"/api/knowledge/mound/graph/{node_id}/related", params=params
        )

    # ========== Sync Operations ==========

    def sync_continuum(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Sync knowledge from ContinuumMemory."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/knowledge/mound/sync/continuum", json=payload)

    def sync_consensus(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Sync knowledge from ConsensusMemory."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/knowledge/mound/sync/consensus", json=payload)

    def sync_facts(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Sync knowledge from FactStore."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/knowledge/mound/sync/facts", json=payload)

    # ========== Federation Operations ==========

    def register_region(
        self,
        region_id: str,
        endpoint: str,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Register a federated region."""
        payload: dict[str, Any] = {"region_id": region_id, "endpoint": endpoint}
        if api_key:
            payload["api_key"] = api_key
        return self._client.request("POST", "/api/knowledge/mound/federation/regions", json=payload)

    def list_regions(self) -> dict[str, Any]:
        """List all federated regions."""
        return self._client.request("GET", "/api/knowledge/mound/federation/regions")

    def unregister_region(self, region_id: str) -> dict[str, Any]:
        """Unregister a federated region."""
        return self._client.request(
            "DELETE", f"/api/knowledge/mound/federation/regions/{region_id}"
        )

    def federation_sync_push(self, region_id: str) -> dict[str, Any]:
        """Push sync to a specific region."""
        return self._client.request(
            "POST", "/api/knowledge/mound/federation/sync/push", json={"region_id": region_id}
        )

    def federation_sync_pull(self, region_id: str) -> dict[str, Any]:
        """Pull sync from a specific region."""
        return self._client.request(
            "POST", "/api/knowledge/mound/federation/sync/pull", json={"region_id": region_id}
        )

    def federation_sync_all(self) -> dict[str, Any]:
        """Sync with all federated regions."""
        return self._client.request("POST", "/api/knowledge/mound/federation/sync/all", json={})

    def federation_status(self) -> dict[str, Any]:
        """Get federation status."""
        return self._client.request("GET", "/api/knowledge/mound/federation/status")

    # ========== Export Operations ==========

    def export_d3(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Export knowledge graph as D3 JSON."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/knowledge/mound/export/d3", params=params)

    def export_graphml(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Export knowledge graph as GraphML."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/knowledge/mound/export/graphml", params=params)

    # ========== Contradiction Detection ==========

    def detect_contradictions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Trigger contradiction detection scan."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return self._client.request(
            "POST", "/api/knowledge/mound/contradictions/detect", json=payload
        )

    def list_mound_contradictions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List unresolved contradictions in the mound."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request("GET", "/api/knowledge/mound/contradictions", params=params)

    def resolve_contradiction(
        self,
        contradiction_id: str,
        resolution: str,
        keep_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a contradiction."""
        payload: dict[str, Any] = {"resolution": resolution}
        if keep_node_id:
            payload["keep_node_id"] = keep_node_id
        return self._client.request(
            "POST", f"/api/knowledge/mound/contradictions/{contradiction_id}/resolve", json=payload
        )

    def contradiction_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get contradiction statistics."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "GET", "/api/knowledge/mound/contradictions/stats", params=params
        )


class AsyncKnowledgeAPI:
    """Asynchronous Knowledge Base API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def search(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"q": query, "limit": limit}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/knowledge/search", params=params)

    async def query(self, prompt: str) -> dict[str, Any]:
        return await self._client.request(
            "POST", "/api/v1/knowledge/query", json={"prompt": prompt}
        )

    async def list_facts(
        self,
        workspace_id: str | None = None,
        topic: str | None = None,
        min_confidence: float = 0.0,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
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
        return await self._client.request("GET", "/api/v1/knowledge/facts", params=params)

    def list_all_facts(
        self,
        workspace_id: str | None = None,
        topic: str | None = None,
        min_confidence: float = 0.0,
        status: str | None = None,
        include_superseded: bool = False,
        page_size: int = 50,
    ):
        """
        Iterate through all facts with automatic pagination.

        Args:
            workspace_id: Filter by workspace
            topic: Filter by topic
            min_confidence: Minimum confidence threshold
            status: Filter by status
            include_superseded: Include superseded facts
            page_size: Number of facts per page (default 50)

        Returns:
            AsyncPaginator yielding fact dictionaries

        Example::

            async for fact in client.knowledge.list_all_facts(topic="security"):
                print(fact["statement"])
        """
        from ..pagination import AsyncPaginator

        params: dict[str, Any] = {
            "min_confidence": min_confidence,
            "include_superseded": include_superseded,
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        if topic:
            params["topic"] = topic
        if status:
            params["status"] = status

        return AsyncPaginator(self._client, "/api/v1/knowledge/facts", params, page_size)

    async def get_fact(self, fact_id: str) -> dict[str, Any]:
        return await self._client.request("GET", f"/api/v1/knowledge/facts/{fact_id}")

    async def create_fact(
        self,
        statement: str,
        workspace_id: str = "default",
        confidence: float = 0.5,
        topics: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        source_documents: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        return await self._client.request("POST", "/api/v1/knowledge/facts", json=payload)

    async def update_fact(
        self,
        fact_id: str,
        confidence: float | None = None,
        validation_status: str | None = None,
        evidence_ids: list[str] | None = None,
        topics: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if confidence is not None:
            payload["confidence"] = confidence
        if validation_status is not None:
            payload["validation_status"] = validation_status
        if evidence_ids is not None:
            payload["evidence_ids"] = evidence_ids
        if topics is not None:
            payload["topics"] = topics
        if metadata is not None:
            payload["metadata"] = metadata
        if superseded_by is not None:
            payload["superseded_by"] = superseded_by
        return await self._client.request("PUT", f"/api/v1/knowledge/facts/{fact_id}", json=payload)

    async def delete_fact(self, fact_id: str) -> dict[str, Any]:
        return await self._client.request("DELETE", f"/api/v1/knowledge/facts/{fact_id}")

    async def verify_fact(self, fact_id: str) -> dict[str, Any]:
        return await self._client.request(
            "POST", f"/api/v1/knowledge/facts/{fact_id}/verify", json={}
        )

    async def list_contradictions(self, fact_id: str) -> dict[str, Any]:
        return await self._client.request(
            "GET", f"/api/v1/knowledge/facts/{fact_id}/contradictions"
        )

    async def list_relations(
        self,
        fact_id: str,
        relation_type: str | None = None,
        as_source: bool = True,
        as_target: bool = True,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"as_source": as_source, "as_target": as_target}
        if relation_type:
            params["type"] = relation_type
        return await self._client.request(
            "GET", f"/api/v1/knowledge/facts/{fact_id}/relations", params=params
        )

    async def add_relation(
        self,
        fact_id: str,
        target_fact_id: str,
        relation_type: str,
    ) -> dict[str, Any]:
        payload = {"target_fact_id": target_fact_id, "relation_type": relation_type}
        return await self._client.request(
            "POST", f"/api/v1/knowledge/facts/{fact_id}/relations", json=payload
        )

    async def add_relation_between_facts(
        self,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: str,
        confidence: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_fact_id": source_fact_id,
            "target_fact_id": target_fact_id,
            "relation_type": relation_type,
            "confidence": confidence,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client.request("POST", "/api/v1/knowledge/facts/relations", json=payload)

    async def get_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/knowledge/stats", params=params)

    async def get_mound_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/v1/knowledge/mound/stats", params=params)

    # ========== Knowledge Mound Node Operations ==========

    async def mound_query(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Semantic query against Knowledge Mound."""
        payload: dict[str, Any] = {"query": query, "limit": limit}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/knowledge/mound/query", json=payload)

    async def add_node(
        self,
        content: str,
        node_type: str = "fact",
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Add a knowledge node to the mound."""
        payload: dict[str, Any] = {"content": content, "node_type": node_type}
        if metadata:
            payload["metadata"] = metadata
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/knowledge/mound/nodes", json=payload)

    async def get_node(self, node_id: str) -> dict[str, Any]:
        """Get a specific knowledge node."""
        return await self._client.request("GET", f"/api/knowledge/mound/nodes/{node_id}")

    async def list_nodes(
        self,
        workspace_id: str | None = None,
        node_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List knowledge nodes with filtering."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if workspace_id:
            params["workspace_id"] = workspace_id
        if node_type:
            params["node_type"] = node_type
        return await self._client.request("GET", "/api/knowledge/mound/nodes", params=params)

    async def get_node_relationships(self, node_id: str) -> dict[str, Any]:
        """Get relationships for a knowledge node."""
        return await self._client.request(
            "GET", f"/api/knowledge/mound/nodes/{node_id}/relationships"
        )

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a relationship between nodes."""
        payload: dict[str, Any] = {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
        }
        if metadata:
            payload["metadata"] = metadata
        return await self._client.request(
            "POST", "/api/knowledge/mound/relationships", json=payload
        )

    # ========== Graph Operations ==========

    async def get_graph(
        self,
        node_id: str,
        depth: int = 2,
        include_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get graph traversal from a node."""
        params: dict[str, Any] = {"depth": depth}
        if include_types:
            params["include_types"] = ",".join(include_types)
        return await self._client.request(
            "GET", f"/api/knowledge/mound/graph/{node_id}", params=params
        )

    async def get_lineage(self, node_id: str) -> dict[str, Any]:
        """Get lineage (provenance chain) for a node."""
        return await self._client.request("GET", f"/api/knowledge/mound/graph/{node_id}/lineage")

    async def get_related(
        self,
        node_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get semantically related nodes."""
        params: dict[str, Any] = {"limit": limit}
        return await self._client.request(
            "GET", f"/api/knowledge/mound/graph/{node_id}/related", params=params
        )

    # ========== Sync Operations ==========

    async def sync_continuum(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Sync knowledge from ContinuumMemory."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return await self._client.request(
            "POST", "/api/knowledge/mound/sync/continuum", json=payload
        )

    async def sync_consensus(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Sync knowledge from ConsensusMemory."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return await self._client.request(
            "POST", "/api/knowledge/mound/sync/consensus", json=payload
        )

    async def sync_facts(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Sync knowledge from FactStore."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return await self._client.request("POST", "/api/knowledge/mound/sync/facts", json=payload)

    # ========== Federation Operations ==========

    async def register_region(
        self,
        region_id: str,
        endpoint: str,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Register a federated region."""
        payload: dict[str, Any] = {"region_id": region_id, "endpoint": endpoint}
        if api_key:
            payload["api_key"] = api_key
        return await self._client.request(
            "POST", "/api/knowledge/mound/federation/regions", json=payload
        )

    async def list_regions(self) -> dict[str, Any]:
        """List all federated regions."""
        return await self._client.request("GET", "/api/knowledge/mound/federation/regions")

    async def unregister_region(self, region_id: str) -> dict[str, Any]:
        """Unregister a federated region."""
        return await self._client.request(
            "DELETE", f"/api/knowledge/mound/federation/regions/{region_id}"
        )

    async def federation_sync_push(self, region_id: str) -> dict[str, Any]:
        """Push sync to a specific region."""
        return await self._client.request(
            "POST",
            "/api/knowledge/mound/federation/sync/push",
            json={"region_id": region_id},
        )

    async def federation_sync_pull(self, region_id: str) -> dict[str, Any]:
        """Pull sync from a specific region."""
        return await self._client.request(
            "POST",
            "/api/knowledge/mound/federation/sync/pull",
            json={"region_id": region_id},
        )

    async def federation_sync_all(self) -> dict[str, Any]:
        """Sync with all federated regions."""
        return await self._client.request(
            "POST", "/api/knowledge/mound/federation/sync/all", json={}
        )

    async def federation_status(self) -> dict[str, Any]:
        """Get federation status."""
        return await self._client.request("GET", "/api/knowledge/mound/federation/status")

    # ========== Export Operations ==========

    async def export_d3(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Export knowledge graph as D3 JSON."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request("GET", "/api/knowledge/mound/export/d3", params=params)

    async def export_graphml(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Export knowledge graph as GraphML."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/knowledge/mound/export/graphml", params=params
        )

    # ========== Contradiction Detection ==========

    async def detect_contradictions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Trigger contradiction detection scan."""
        payload: dict[str, Any] = {}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return await self._client.request(
            "POST", "/api/knowledge/mound/contradictions/detect", json=payload
        )

    async def list_mound_contradictions(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List unresolved contradictions in the mound."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/knowledge/mound/contradictions", params=params
        )

    async def resolve_contradiction(
        self,
        contradiction_id: str,
        resolution: str,
        keep_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a contradiction."""
        payload: dict[str, Any] = {"resolution": resolution}
        if keep_node_id:
            payload["keep_node_id"] = keep_node_id
        return await self._client.request(
            "POST",
            f"/api/knowledge/mound/contradictions/{contradiction_id}/resolve",
            json=payload,
        )

    async def contradiction_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get contradiction statistics."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/knowledge/mound/contradictions/stats", params=params
        )

    # ========== Visibility & Access Control ==========

    async def get_visibility(self, node_id: str) -> dict[str, Any]:
        """Get visibility level of a knowledge node."""
        return await self._client.request(
            "GET", f"/api/v1/knowledge/mound/nodes/{node_id}/visibility"
        )

    async def set_visibility(
        self,
        node_id: str,
        visibility: str,  # 'private' | 'workspace' | 'shared' | 'public'
    ) -> dict[str, Any]:
        """Set visibility level of a knowledge node."""
        return await self._client.request(
            "PUT",
            f"/api/v1/knowledge/mound/nodes/{node_id}/visibility",
            json={"visibility": visibility},
        )

    async def list_access_grants(self, node_id: str) -> dict[str, Any]:
        """List access grants for a knowledge node."""
        return await self._client.request("GET", f"/api/v1/knowledge/mound/nodes/{node_id}/access")

    async def grant_access(
        self,
        node_id: str,
        grantee_id: str,
        grantee_type: str,  # 'user' | 'workspace'
        permission: str = "read",  # 'read' | 'write' | 'admin'
    ) -> dict[str, Any]:
        """Grant access to a knowledge node."""
        return await self._client.request(
            "POST",
            f"/api/v1/knowledge/mound/nodes/{node_id}/access",
            json={"grantee_id": grantee_id, "grantee_type": grantee_type, "permission": permission},
        )

    async def revoke_access(self, node_id: str, grant_id: str) -> dict[str, Any]:
        """Revoke access from a knowledge node."""
        return await self._client.request(
            "DELETE",
            f"/api/v1/knowledge/mound/nodes/{node_id}/access",
            json={"grant_id": grant_id},
        )

    # ========== Sharing ==========

    async def share(
        self,
        item_id: str,
        target_id: str,
        target_type: str,  # 'user' | 'workspace'
        permission: str = "read",
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Share a knowledge item with another workspace or user."""
        payload: dict[str, Any] = {
            "item_id": item_id,
            "target_id": target_id,
            "target_type": target_type,
            "permission": permission,
        }
        if expires_at:
            payload["expires_at"] = expires_at
        return await self._client.request("POST", "/api/v1/knowledge/mound/share", json=payload)

    async def get_shared_with_me(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get items shared with the current user/workspace."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/shared-with-me",
            params={"limit": limit, "offset": offset},
        )

    async def get_my_shares(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get items shared by the current user."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/my-shares",
            params={"limit": limit, "offset": offset},
        )

    async def revoke_share(self, share_id: str) -> dict[str, Any]:
        """Revoke a share."""
        return await self._client.request(
            "DELETE", "/api/v1/knowledge/mound/share", json={"share_id": share_id}
        )

    # ========== Global Knowledge ==========

    async def store_global_fact(
        self,
        content: str,
        source: str,
        confidence: float,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Store a verified fact as global knowledge (admin only)."""
        payload: dict[str, Any] = {
            "content": content,
            "source": source,
            "confidence": confidence,
        }
        if tags:
            payload["tags"] = tags
        return await self._client.request("POST", "/api/v1/knowledge/mound/global", json=payload)

    async def query_global(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Query global/system knowledge."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/global",
            params={"query": query, "limit": limit},
        )

    async def promote_to_global(self, node_id: str, review_required: bool = True) -> dict[str, Any]:
        """Promote workspace knowledge to global level."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/global/promote",
            json={"node_id": node_id, "review_required": review_required},
        )

    async def get_system_facts(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """List all system-level verified facts."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/global/facts",
            params={"limit": limit, "offset": offset},
        )

    # ========== Deduplication ==========

    async def get_duplicate_clusters(
        self, threshold: float = 0.9, limit: int = 50
    ) -> dict[str, Any]:
        """Find duplicate clusters by similarity threshold."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/dedup/clusters",
            params={"threshold": threshold, "limit": limit},
        )

    async def get_dedup_report(self) -> dict[str, Any]:
        """Generate deduplication analysis report."""
        return await self._client.request("GET", "/api/v1/knowledge/mound/dedup/report")

    async def merge_duplicate_cluster(
        self,
        cluster_id: str,
        primary_id: str | None = None,
        strategy: str = "highest_confidence",
    ) -> dict[str, Any]:
        """Merge a specific duplicate cluster."""
        payload: dict[str, Any] = {"cluster_id": cluster_id, "strategy": strategy}
        if primary_id:
            payload["primary_id"] = primary_id
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/dedup/merge", json=payload
        )

    async def auto_merge_exact_duplicates(self, dry_run: bool = False) -> dict[str, Any]:
        """Automatically merge exact duplicates."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/dedup/auto-merge",
            json={"dry_run": dry_run},
        )

    # ========== Pruning ==========

    async def get_prunable_items(
        self,
        max_age_days: int | None = None,
        min_staleness: float | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get items eligible for pruning by staleness/age."""
        params: dict[str, Any] = {"limit": limit}
        if max_age_days:
            params["max_age_days"] = max_age_days
        if min_staleness:
            params["min_staleness"] = min_staleness
        return await self._client.request(
            "GET", "/api/v1/knowledge/mound/pruning/items", params=params
        )

    async def execute_prune(self, node_ids: list[str], action: str = "archive") -> dict[str, Any]:
        """Prune specified items (archive or delete)."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/pruning/execute",
            json={"node_ids": node_ids, "action": action},
        )

    async def auto_prune(self, policy: str = "moderate", dry_run: bool = False) -> dict[str, Any]:
        """Run auto-prune with policy."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/pruning/auto",
            json={"policy": policy, "dry_run": dry_run},
        )

    async def get_prune_history(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get pruning history."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/pruning/history",
            params={"limit": limit, "offset": offset},
        )

    async def restore_pruned_item(self, node_id: str) -> dict[str, Any]:
        """Restore an archived item."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/pruning/restore",
            json={"node_id": node_id},
        )

    # ========== Culture ==========

    async def get_culture(self) -> dict[str, Any]:
        """Get organization culture profile."""
        return await self._client.request("GET", "/api/v1/knowledge/mound/culture")

    async def add_culture_document(
        self,
        doc_type: str,  # 'policy' | 'principle' | 'value'
        content: str,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Add a culture document."""
        payload: dict[str, Any] = {"type": doc_type, "content": content}
        if source:
            payload["source"] = source
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/culture/documents", json=payload
        )

    async def promote_to_culture(self, node_id: str, doc_type: str | None = None) -> dict[str, Any]:
        """Promote knowledge to culture level."""
        payload: dict[str, Any] = {"node_id": node_id}
        if doc_type:
            payload["type"] = doc_type
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/culture/promote", json=payload
        )

    # ========== Dashboard & Metrics ==========

    async def get_dashboard_health(self) -> dict[str, Any]:
        """Get Knowledge Mound health status and recommendations."""
        return await self._client.request("GET", "/api/v1/knowledge/mound/dashboard/health")

    async def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get detailed operational metrics."""
        return await self._client.request("GET", "/api/v1/knowledge/mound/dashboard/metrics")

    async def reset_dashboard_metrics(self) -> dict[str, Any]:
        """Reset metrics counters."""
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/dashboard/metrics/reset", json={}
        )

    async def get_dashboard_adapters(self) -> dict[str, Any]:
        """Get adapter status and health."""
        return await self._client.request("GET", "/api/v1/knowledge/mound/dashboard/adapters")

    # ========== Governance ==========

    async def create_governance_role(
        self,
        name: str,
        permissions: list[str],
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a custom role for knowledge governance."""
        payload: dict[str, Any] = {"name": name, "permissions": permissions}
        if description:
            payload["description"] = description
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/governance/roles", json=payload
        )

    async def assign_governance_role(self, user_id: str, role_id: str) -> dict[str, Any]:
        """Assign a role to a user."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/governance/roles/assign",
            json={"user_id": user_id, "role_id": role_id},
        )

    async def revoke_governance_role(self, user_id: str, role_id: str) -> dict[str, Any]:
        """Revoke a role from a user."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/governance/roles/revoke",
            json={"user_id": user_id, "role_id": role_id},
        )

    async def get_user_governance_permissions(self, user_id: str) -> dict[str, Any]:
        """Get user permissions for knowledge governance."""
        return await self._client.request(
            "GET", f"/api/v1/knowledge/mound/governance/permissions/{user_id}"
        )

    async def check_governance_permission(self, user_id: str, permission: str) -> dict[str, Any]:
        """Check if user has a specific permission."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/governance/permissions/check",
            json={"user_id": user_id, "permission": permission},
        )

    async def query_governance_audit(
        self,
        user_id: str | None = None,
        action: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query the governance audit trail."""
        params: dict[str, Any] = {"limit": limit}
        if user_id:
            params["user_id"] = user_id
        if action:
            params["action"] = action
        if since:
            params["since"] = since
        return await self._client.request(
            "GET", "/api/v1/knowledge/mound/governance/audit", params=params
        )

    # ========== Analytics ==========

    async def analyze_coverage(self, topics: list[str] | None = None) -> dict[str, Any]:
        """Analyze domain coverage by topic."""
        params: dict[str, Any] = {}
        if topics:
            params["topics"] = ",".join(topics)
        return await self._client.request(
            "GET", "/api/v1/knowledge/mound/analytics/coverage", params=params
        )

    async def analyze_usage(self, period: str = "week", since: str | None = None) -> dict[str, Any]:
        """Analyze usage patterns over time."""
        params: dict[str, Any] = {"period": period}
        if since:
            params["since"] = since
        return await self._client.request(
            "GET", "/api/v1/knowledge/mound/analytics/usage", params=params
        )

    async def record_usage_event(
        self,
        node_id: str,
        event_type: str,  # 'query' | 'view' | 'cite' | 'share' | 'export'
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a usage event."""
        payload: dict[str, Any] = {"node_id": node_id, "event_type": event_type}
        if metadata:
            payload["metadata"] = metadata
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/analytics/usage/record", json=payload
        )

    async def capture_quality_snapshot(self) -> dict[str, Any]:
        """Capture quality metrics snapshot."""
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/analytics/quality/snapshot", json={}
        )

    async def get_quality_trend(
        self, period: str = "week", metrics: list[str] | None = None
    ) -> dict[str, Any]:
        """Get quality metrics trend over time."""
        params: dict[str, Any] = {"period": period}
        if metrics:
            params["metrics"] = ",".join(metrics)
        return await self._client.request(
            "GET", "/api/v1/knowledge/mound/analytics/quality/trend", params=params
        )

    # ========== Extraction ==========

    async def extract_from_debate(
        self,
        debate_id: str,
        confidence_threshold: float = 0.7,
        auto_promote: bool = False,
    ) -> dict[str, Any]:
        """Extract claims/knowledge from a debate."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/extraction/debate",
            json={
                "debate_id": debate_id,
                "confidence_threshold": confidence_threshold,
                "auto_promote": auto_promote,
            },
        )

    async def promote_extracted(
        self, claim_ids: list[str], target_tier: str | None = None
    ) -> dict[str, Any]:
        """Promote extracted claims to main knowledge."""
        payload: dict[str, Any] = {"claim_ids": claim_ids}
        if target_tier:
            payload["target_tier"] = target_tier
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/extraction/promote", json=payload
        )

    # ========== Confidence Decay ==========

    async def apply_confidence_decay(
        self, scope: str = "workspace", decay_rate: float | None = None
    ) -> dict[str, Any]:
        """Apply confidence decay to workspace knowledge."""
        payload: dict[str, Any] = {"scope": scope}
        if decay_rate:
            payload["decay_rate"] = decay_rate
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/confidence/decay", json=payload
        )

    async def record_confidence_event(
        self,
        node_id: str,
        event_type: str,  # 'validation' | 'contradiction' | 'citation' | 'correction'
        impact: float,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Record a confidence-affecting event."""
        payload: dict[str, Any] = {
            "node_id": node_id,
            "type": event_type,
            "impact": impact,
        }
        if notes:
            payload["notes"] = notes
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/confidence/event", json=payload
        )

    async def get_confidence_history(
        self, node_id: str, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """Get confidence adjustment history for a node."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/confidence/history",
            params={"node_id": node_id, "limit": limit, "offset": offset},
        )

    # ========== Auto-Curation ==========

    async def get_curation_policy(self) -> dict[str, Any]:
        """Get curation policy for workspace."""
        return await self._client.request("GET", "/api/v1/knowledge/mound/curation/policy")

    async def set_curation_policy(
        self,
        auto_promote: bool | None = None,
        auto_archive_days: int | None = None,
        quality_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Set curation policy."""
        payload: dict[str, Any] = {}
        if auto_promote is not None:
            payload["auto_promote"] = auto_promote
        if auto_archive_days is not None:
            payload["auto_archive_days"] = auto_archive_days
        if quality_threshold is not None:
            payload["quality_threshold"] = quality_threshold
        return await self._client.request(
            "POST", "/api/v1/knowledge/mound/curation/policy", json=payload
        )

    async def get_curation_status(self) -> dict[str, Any]:
        """Get curation status."""
        return await self._client.request("GET", "/api/v1/knowledge/mound/curation/status")

    async def run_curation(self, dry_run: bool = False) -> dict[str, Any]:
        """Trigger a curation run."""
        return await self._client.request(
            "POST",
            "/api/v1/knowledge/mound/curation/run",
            json={"dry_run": dry_run},
        )

    async def get_curation_history(self, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        """Get curation history."""
        return await self._client.request(
            "GET",
            "/api/v1/knowledge/mound/curation/history",
            params={"limit": limit, "offset": offset},
        )
