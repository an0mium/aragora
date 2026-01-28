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
