"""
Knowledge API resource for the Aragora client.

Provides methods for interacting with the Knowledge Mound:
- Search and query knowledge
- Create, update, delete knowledge nodes
- Analytics and governance
- Contradiction detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """A knowledge node in the Knowledge Mound."""

    id: str
    content: str
    node_type: str
    confidence: float = 0.8
    workspace_id: str | None = None
    domain: str | None = None
    source_debate_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class KnowledgeSearchResult:
    """A search result from the Knowledge Mound."""

    node_id: str
    content: str
    score: float
    node_type: str
    confidence: float
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeStats:
    """Statistics about the Knowledge Mound."""

    total_nodes: int
    nodes_by_type: dict[str, int]
    nodes_by_tier: dict[str, int]
    total_relationships: int
    average_confidence: float
    stale_nodes_count: int


@dataclass
class CoverageReport:
    """Domain coverage analysis report."""

    domains: dict[str, int]
    coverage_score: float
    gaps: list[str]
    recommendations: list[str]


@dataclass
class ContradictionResult:
    """A detected contradiction in knowledge."""

    id: str
    node_a_id: str
    node_b_id: str
    contradiction_type: str
    severity: str
    description: str
    suggested_resolution: str | None = None


class KnowledgeAPI:
    """API interface for Knowledge Mound operations."""

    def __init__(self, client: AragoraClient):
        self._client = client

    # =========================================================================
    # Search and Query
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str | None = None,
        domain: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[KnowledgeSearchResult]:
        """
        Search the knowledge base.

        Args:
            query: Search query text
            limit: Maximum results to return
            workspace_id: Filter by workspace
            domain: Filter by domain
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching knowledge entries
        """
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        if domain:
            params["domain"] = domain
        if min_confidence > 0:
            params["min_confidence"] = min_confidence

        response = self._client._get("/api/v1/knowledge/search", params)
        results = response.get("results", [])
        return [KnowledgeSearchResult(**r) for r in results]

    async def search_async(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str | None = None,
        domain: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[KnowledgeSearchResult]:
        """Async version of search()."""
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if workspace_id:
            params["workspace_id"] = workspace_id
        if domain:
            params["domain"] = domain
        if min_confidence > 0:
            params["min_confidence"] = min_confidence

        response = await self._client._get_async("/api/v1/knowledge/search", params)
        results = response.get("results", [])
        return [KnowledgeSearchResult(**r) for r in results]

    def semantic_query(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search on the Knowledge Mound.

        Args:
            query: Natural language query
            limit: Maximum results
            workspace_id: Filter by workspace

        Returns:
            List of semantically relevant knowledge items
        """
        body = {
            "query": query,
            "limit": limit,
        }
        if workspace_id:
            body["workspace_id"] = workspace_id

        response = self._client._post("/api/v1/knowledge/mound/query", body)
        return response.get("results", [])

    async def semantic_query_async(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async version of semantic_query()."""
        body = {
            "query": query,
            "limit": limit,
        }
        if workspace_id:
            body["workspace_id"] = workspace_id

        response = await self._client._post_async("/api/v1/knowledge/mound/query", body)
        return response.get("results", [])

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create_node(
        self,
        content: str,
        node_type: str = "fact",
        confidence: float = 0.8,
        workspace_id: str | None = None,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode:
        """
        Create a new knowledge node.

        Args:
            content: The knowledge content
            node_type: Type of node (fact, claim, evidence, etc.)
            confidence: Confidence score (0.0-1.0)
            workspace_id: Workspace to add to
            domain: Knowledge domain
            metadata: Additional metadata

        Returns:
            Created KnowledgeNode
        """
        body = {
            "content": content,
            "node_type": node_type,
            "confidence": confidence,
        }
        if workspace_id:
            body["workspace_id"] = workspace_id
        if domain:
            body["domain"] = domain
        if metadata:
            body["metadata"] = metadata

        response = self._client._post("/api/v1/knowledge/mound/nodes", body)
        return KnowledgeNode(**response)

    async def create_node_async(
        self,
        content: str,
        node_type: str = "fact",
        confidence: float = 0.8,
        workspace_id: str | None = None,
        domain: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode:
        """Async version of create_node()."""
        body = {
            "content": content,
            "node_type": node_type,
            "confidence": confidence,
        }
        if workspace_id:
            body["workspace_id"] = workspace_id
        if domain:
            body["domain"] = domain
        if metadata:
            body["metadata"] = metadata

        response = await self._client._post_async("/api/v1/knowledge/mound/nodes", body)
        return KnowledgeNode(**response)

    def get_node(self, node_id: str) -> KnowledgeNode:
        """Get a knowledge node by ID."""
        response = self._client._get(f"/api/v1/knowledge/{node_id}")
        return KnowledgeNode(**response)

    async def get_node_async(self, node_id: str) -> KnowledgeNode:
        """Async version of get_node()."""
        response = await self._client._get_async(f"/api/v1/knowledge/{node_id}")
        return KnowledgeNode(**response)

    def update_node(
        self,
        node_id: str,
        content: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode:
        """Update a knowledge node."""
        body: dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        if confidence is not None:
            body["confidence"] = confidence
        if metadata is not None:
            body["metadata"] = metadata

        response = self._client._patch(f"/api/v1/knowledge/{node_id}", body)
        return KnowledgeNode(**response)

    async def update_node_async(
        self,
        node_id: str,
        content: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode:
        """Async version of update_node()."""
        body: dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        if confidence is not None:
            body["confidence"] = confidence
        if metadata is not None:
            body["metadata"] = metadata

        response = await self._client._patch_async(f"/api/v1/knowledge/{node_id}", body)
        return KnowledgeNode(**response)

    def delete_node(self, node_id: str) -> bool:
        """Delete a knowledge node."""
        response = self._client._delete(f"/api/v1/knowledge/{node_id}")
        return response.get("deleted", False)

    async def delete_node_async(self, node_id: str) -> bool:
        """Async version of delete_node()."""
        response = await self._client._delete_async(f"/api/v1/knowledge/{node_id}")
        return response.get("deleted", False)

    # =========================================================================
    # Statistics and Analytics
    # =========================================================================

    def get_stats(self, workspace_id: str | None = None) -> KnowledgeStats:
        """Get Knowledge Mound statistics."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        response = self._client._get("/api/v1/knowledge/mound/stats", params)
        return KnowledgeStats(**response)

    async def get_stats_async(self, workspace_id: str | None = None) -> KnowledgeStats:
        """Async version of get_stats()."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        response = await self._client._get_async("/api/v1/knowledge/mound/stats", params)
        return KnowledgeStats(**response)

    def get_coverage_analytics(self, workspace_id: str | None = None) -> CoverageReport:
        """Get domain coverage analytics."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        response = self._client._get("/api/v1/knowledge/mound/analytics/coverage", params)
        return CoverageReport(**response)

    async def get_coverage_analytics_async(self, workspace_id: str | None = None) -> CoverageReport:
        """Async version of get_coverage_analytics()."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        response = await self._client._get_async(
            "/api/v1/knowledge/mound/analytics/coverage", params
        )
        return CoverageReport(**response)

    def get_usage_analytics(
        self, workspace_id: str | None = None, days: int = 30
    ) -> dict[str, Any]:
        """Get usage analytics."""
        params: dict[str, Any] = {"days": days}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client._get("/api/v1/knowledge/mound/analytics/usage", params)

    async def get_usage_analytics_async(
        self, workspace_id: str | None = None, days: int = 30
    ) -> dict[str, Any]:
        """Async version of get_usage_analytics()."""
        params: dict[str, Any] = {"days": days}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client._get_async("/api/v1/knowledge/mound/analytics/usage", params)

    def get_quality_snapshot(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Capture current quality metrics snapshot."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        return self._client._get("/api/v1/knowledge/mound/analytics/quality/snapshot", params)

    async def get_quality_snapshot_async(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Async version of get_quality_snapshot()."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        return await self._client._get_async(
            "/api/v1/knowledge/mound/analytics/quality/snapshot", params
        )

    # =========================================================================
    # Health and Dashboard
    # =========================================================================

    def get_health(self) -> dict[str, Any]:
        """Get Knowledge Mound health status."""
        return self._client._get("/api/v1/knowledge/mound/dashboard/health")

    async def get_health_async(self) -> dict[str, Any]:
        """Async version of get_health()."""
        return await self._client._get_async("/api/v1/knowledge/mound/dashboard/health")

    def get_metrics(self) -> dict[str, Any]:
        """Get Knowledge Mound metrics."""
        return self._client._get("/api/v1/knowledge/mound/dashboard/metrics")

    async def get_metrics_async(self) -> dict[str, Any]:
        """Async version of get_metrics()."""
        return await self._client._get_async("/api/v1/knowledge/mound/dashboard/metrics")

    def get_adapters(self) -> list[dict[str, Any]]:
        """Get Knowledge Mound adapter status."""
        response = self._client._get("/api/v1/knowledge/mound/dashboard/adapters")
        return response.get("adapters", [])

    async def get_adapters_async(self) -> list[dict[str, Any]]:
        """Async version of get_adapters()."""
        response = await self._client._get_async("/api/v1/knowledge/mound/dashboard/adapters")
        return response.get("adapters", [])

    # =========================================================================
    # Contradiction Detection
    # =========================================================================

    def detect_contradictions(
        self,
        workspace_id: str | None = None,
        threshold: float = 0.7,
    ) -> list[ContradictionResult]:
        """
        Detect contradictions in the knowledge base.

        Args:
            workspace_id: Workspace to analyze
            threshold: Similarity threshold for detection

        Returns:
            List of detected contradictions
        """
        body: dict[str, Any] = {"threshold": threshold}
        if workspace_id:
            body["workspace_id"] = workspace_id

        response = self._client._post("/api/v1/knowledge/mound/contradictions/detect", body)
        contradictions = response.get("contradictions", [])
        return [ContradictionResult(**c) for c in contradictions]

    async def detect_contradictions_async(
        self,
        workspace_id: str | None = None,
        threshold: float = 0.7,
    ) -> list[ContradictionResult]:
        """Async version of detect_contradictions()."""
        body: dict[str, Any] = {"threshold": threshold}
        if workspace_id:
            body["workspace_id"] = workspace_id

        response = await self._client._post_async(
            "/api/v1/knowledge/mound/contradictions/detect", body
        )
        contradictions = response.get("contradictions", [])
        return [ContradictionResult(**c) for c in contradictions]

    def resolve_contradiction(
        self,
        contradiction_id: str,
        resolution: str,
        keep_node_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Resolve a knowledge contradiction.

        Args:
            contradiction_id: ID of the contradiction
            resolution: Resolution strategy (keep_a, keep_b, merge, deprecate_both)
            keep_node_id: Which node to keep (if applicable)

        Returns:
            Resolution result
        """
        body = {"resolution": resolution}
        if keep_node_id:
            body["keep_node_id"] = keep_node_id

        return self._client._post(
            f"/api/v1/knowledge/mound/contradictions/{contradiction_id}/resolve", body
        )

    async def resolve_contradiction_async(
        self,
        contradiction_id: str,
        resolution: str,
        keep_node_id: str | None = None,
    ) -> dict[str, Any]:
        """Async version of resolve_contradiction()."""
        body = {"resolution": resolution}
        if keep_node_id:
            body["keep_node_id"] = keep_node_id

        return await self._client._post_async(
            f"/api/v1/knowledge/mound/contradictions/{contradiction_id}/resolve", body
        )

    # =========================================================================
    # Knowledge Extraction
    # =========================================================================

    def extract_from_debate(
        self,
        debate_id: str,
        auto_promote: bool = False,
    ) -> dict[str, Any]:
        """
        Extract knowledge from a completed debate.

        Args:
            debate_id: The debate to extract from
            auto_promote: Whether to automatically promote to permanent storage

        Returns:
            Extraction result with extracted items
        """
        body = {
            "debate_id": debate_id,
            "auto_promote": auto_promote,
        }
        return self._client._post("/api/v1/knowledge/mound/extraction/debate", body)

    async def extract_from_debate_async(
        self,
        debate_id: str,
        auto_promote: bool = False,
    ) -> dict[str, Any]:
        """Async version of extract_from_debate()."""
        body = {
            "debate_id": debate_id,
            "auto_promote": auto_promote,
        }
        return await self._client._post_async("/api/v1/knowledge/mound/extraction/debate", body)

    def promote_extracted(self, extraction_ids: list[str]) -> dict[str, Any]:
        """Promote extracted knowledge to permanent storage."""
        body = {"extraction_ids": extraction_ids}
        return self._client._post("/api/v1/knowledge/mound/extraction/promote", body)

    async def promote_extracted_async(self, extraction_ids: list[str]) -> dict[str, Any]:
        """Async version of promote_extracted()."""
        body = {"extraction_ids": extraction_ids}
        return await self._client._post_async("/api/v1/knowledge/mound/extraction/promote", body)


__all__ = [
    "KnowledgeAPI",
    "KnowledgeNode",
    "KnowledgeSearchResult",
    "KnowledgeStats",
    "CoverageReport",
    "ContradictionResult",
]
