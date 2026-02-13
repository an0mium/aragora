"""
Knowledge Namespace API

Provides methods for interacting with the Knowledge Base:
- Fact listing and creation
- Natural language queries and semantic search
- Knowledge Mound operations (nodes, graph, contradictions, dedup, pruning)
- Governance, analytics, extraction, confidence, curation
- Dashboard and export operations

Note: Fact-specific CRUD (get/update/delete by ID, verify, relations),
federation, sync, sharing, visibility/access, global knowledge, and
culture document operations were removed as their handler routes
no longer exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient
    from ..pagination import AsyncPaginator, SyncPaginator


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
    ) -> SyncPaginator:
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
    ) -> AsyncPaginator:
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
