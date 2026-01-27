"""
Deduplication operations mixin for Knowledge Mound handler.

Provides HTTP endpoints for finding and merging duplicate knowledge items:
- GET /api/knowledge/mound/dedup/clusters - Find duplicate clusters
- GET /api/knowledge/mound/dedup/report - Generate dedup report
- POST /api/knowledge/mound/dedup/merge - Merge a duplicate cluster
- POST /api/knowledge/mound/dedup/auto-merge - Auto-merge exact duplicates
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from aragora.rbac.decorators import require_permission

from ...base import (
    HandlerResult,
    error_response,
    json_response,
    safe_error_message,
)
from ...utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


class DedupOperationsMixin:
    """Mixin providing deduplication API endpoints."""

    ctx: Dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound instance."""
        raise NotImplementedError("Subclass must implement _get_mound")

    @rate_limit(requests_per_minute=30)
    @require_permission("knowledge:read")
    async def get_duplicate_clusters(
        self,
        workspace_id: str,
        similarity_threshold: float = 0.9,
        limit: int = 100,
    ) -> HandlerResult:
        """
        Find duplicate clusters in the workspace.

        GET /api/knowledge/mound/dedup/clusters?workspace_id=...&similarity_threshold=0.9

        Args:
            workspace_id: Workspace to analyze
            similarity_threshold: Minimum similarity to consider duplicates (0.0-1.0)
            limit: Maximum clusters to return

        Returns:
            List of DuplicateCluster objects
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            clusters = await mound.find_duplicates(
                workspace_id=workspace_id,
                similarity_threshold=similarity_threshold,
                limit=limit,
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "similarity_threshold": similarity_threshold,
                    "clusters_found": len(clusters),
                    "clusters": [
                        {
                            "cluster_id": c.cluster_id,
                            "primary_node_id": c.primary_node_id,
                            "duplicate_count": len(c.duplicates),
                            "avg_similarity": c.avg_similarity,
                            "recommended_action": c.recommended_action,
                            "duplicates": [
                                {
                                    "node_id": d.node_id,
                                    "similarity": d.similarity,
                                    "content_preview": d.content_preview,
                                    "tier": d.tier,
                                    "confidence": d.confidence,
                                }
                                for d in c.duplicates
                            ],
                        }
                        for c in clusters
                    ],
                }
            )
        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=10)
    @require_permission("knowledge:read")
    async def get_dedup_report(
        self,
        workspace_id: str,
        similarity_threshold: float = 0.9,
    ) -> HandlerResult:
        """
        Generate comprehensive deduplication report.

        GET /api/knowledge/mound/dedup/report?workspace_id=...

        Args:
            workspace_id: Workspace to analyze

        Returns:
            DedupReport with statistics and recommendations
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            report = await mound.generate_dedup_report(
                workspace_id=workspace_id,
                similarity_threshold=similarity_threshold,
            )

            return json_response(
                {
                    "workspace_id": report.workspace_id,
                    "generated_at": report.generated_at.isoformat(),
                    "total_nodes_analyzed": report.total_nodes_analyzed,
                    "duplicate_clusters_found": report.duplicate_clusters_found,
                    "estimated_reduction_percent": report.estimated_reduction_percent,
                    "cluster_count": len(report.clusters),
                }
            )
        except Exception as e:
            logger.error(f"Error generating dedup report: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=20)
    @require_permission("knowledge:read")
    async def merge_duplicate_cluster(
        self,
        workspace_id: str,
        cluster_id: str,
        primary_node_id: Optional[str] = None,
        archive: bool = True,
    ) -> HandlerResult:
        """
        Merge a duplicate cluster.

        POST /api/knowledge/mound/dedup/merge
        {
            "workspace_id": "...",
            "cluster_id": "...",
            "primary_node_id": "...",  // optional
            "archive": true
        }

        Args:
            workspace_id: Workspace containing the cluster
            cluster_id: Cluster to merge
            primary_node_id: Which node to keep as primary (optional, auto-selected)
            archive: Whether to archive (true) or delete (false) duplicates

        Returns:
            MergeResult with details
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id or not cluster_id:
            return error_response("workspace_id and cluster_id are required", status=400)

        try:
            result = await mound.merge_duplicates(
                workspace_id=workspace_id,
                cluster_id=cluster_id,
                primary_node_id=primary_node_id,
                archive=archive,
            )

            return json_response(
                {
                    "success": True,
                    "kept_node_id": result.kept_node_id,
                    "merged_node_ids": result.merged_node_ids,
                    "archived_count": result.archived_count,
                    "updated_relationships": result.updated_relationships,
                }
            )
        except Exception as e:
            logger.error(f"Error merging duplicates: {e}")
            return error_response(safe_error_message(e), status=500)

    @rate_limit(requests_per_minute=5)
    @require_permission("knowledge:read")
    async def auto_merge_exact_duplicates(
        self,
        workspace_id: str,
        dry_run: bool = True,
    ) -> HandlerResult:
        """
        Automatically merge exact duplicates (content hash match).

        POST /api/knowledge/mound/dedup/auto-merge
        {
            "workspace_id": "...",
            "dry_run": true
        }

        Args:
            workspace_id: Workspace to process
            dry_run: If true, report only without making changes

        Returns:
            Summary of merges performed (or would be performed)
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            result = await mound.auto_merge_exact_duplicates(
                workspace_id=workspace_id,
                dry_run=dry_run,
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "dry_run": result.get("dry_run", dry_run),
                    "duplicates_found": result.get("duplicates_found", 0),
                    "merges_performed": result.get("merges_performed", 0),
                    "details": result.get("details", []),
                }
            )
        except Exception as e:
            logger.error(f"Error in auto-merge: {e}")
            return error_response(safe_error_message(e), status=500)
