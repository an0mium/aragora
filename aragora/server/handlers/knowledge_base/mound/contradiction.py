"""
Contradiction detection operations mixin for Knowledge Mound handler.

Provides HTTP endpoints for detecting and managing knowledge contradictions:
- POST /api/knowledge/mound/contradictions/detect - Trigger contradiction scan
- GET /api/knowledge/mound/contradictions - List unresolved contradictions
- POST /api/knowledge/mound/contradictions/:id/resolve - Resolve a contradiction
- GET /api/knowledge/mound/contradictions/stats - Get contradiction statistics

Phase A2 - Knowledge Quality Assurance
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


class ContradictionOperationsMixin:
    """Mixin providing contradiction detection API endpoints."""

    ctx: Dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound instance."""
        raise NotImplementedError("Subclass must implement _get_mound")

    @require_permission("knowledge:read")
    @rate_limit(requests_per_minute=10)
    async def detect_contradictions(
        self,
        workspace_id: str,
        item_ids: Optional[list[str]] = None,
    ) -> HandlerResult:
        """
        Trigger contradiction detection scan.

        POST /api/knowledge/mound/contradictions/detect
        {
            "workspace_id": "...",
            "item_ids": ["...", "..."]  // optional, scans all if omitted
        }

        Args:
            workspace_id: Workspace to scan
            item_ids: Optional specific item IDs to check

        Returns:
            ContradictionReport with detected contradictions
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            report = await mound.detect_contradictions(
                workspace_id=workspace_id,
                item_ids=item_ids,
            )

            return json_response(report.to_dict())
        except Exception as e:
            logger.error(f"Error detecting contradictions: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:read")
    @rate_limit(requests_per_minute=30)
    async def list_contradictions(
        self,
        workspace_id: Optional[str] = None,
        min_severity: Optional[str] = None,
    ) -> HandlerResult:
        """
        List unresolved contradictions.

        GET /api/knowledge/mound/contradictions?workspace_id=...&min_severity=high

        Args:
            workspace_id: Optional workspace filter
            min_severity: Optional minimum severity (low, medium, high, critical)

        Returns:
            List of unresolved contradictions
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        try:
            contradictions = await mound.get_unresolved_contradictions(
                workspace_id=workspace_id,
                min_severity=min_severity,
            )

            return json_response(
                {
                    "workspace_id": workspace_id,
                    "min_severity": min_severity,
                    "count": len(contradictions),
                    "contradictions": [c.to_dict() for c in contradictions],
                }
            )
        except Exception as e:
            logger.error(f"Error listing contradictions: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:read")
    @rate_limit(requests_per_minute=20)
    async def resolve_contradiction(
        self,
        contradiction_id: str,
        strategy: str,
        resolved_by: Optional[str] = None,
        notes: str = "",
    ) -> HandlerResult:
        """
        Resolve a detected contradiction.

        POST /api/knowledge/mound/contradictions/:id/resolve
        {
            "strategy": "prefer_newer",  // prefer_newer, prefer_higher_confidence, merge, human_review, keep_both
            "resolved_by": "user_id",
            "notes": "Resolution notes..."
        }

        Args:
            contradiction_id: ID of contradiction to resolve
            strategy: Resolution strategy to apply
            resolved_by: User who resolved it
            notes: Optional resolution notes

        Returns:
            Updated contradiction record
        """
        from aragora.knowledge.mound.ops.contradiction import ResolutionStrategy

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not contradiction_id:
            return error_response("contradiction_id is required", status=400)

        if not strategy:
            return error_response("strategy is required", status=400)

        # Validate strategy
        try:
            resolution_strategy = ResolutionStrategy(strategy)
        except ValueError:
            valid_strategies = [s.value for s in ResolutionStrategy]
            return error_response(
                f"Invalid strategy. Must be one of: {valid_strategies}", status=400
            )

        try:
            result = await mound.resolve_contradiction(
                contradiction_id=contradiction_id,
                strategy=resolution_strategy,
                resolved_by=resolved_by,
                notes=notes,
            )

            if result is None:
                return error_response("Contradiction not found", status=404)

            return json_response(
                {
                    "success": True,
                    "contradiction": result.to_dict(),
                }
            )
        except Exception as e:
            logger.error(f"Error resolving contradiction: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("knowledge:read")
    @rate_limit(requests_per_minute=60)
    async def get_contradiction_stats(self) -> HandlerResult:
        """
        Get contradiction detection statistics.

        GET /api/knowledge/mound/contradictions/stats

        Returns:
            Statistics about contradictions
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        try:
            stats = mound.get_contradiction_stats()
            return json_response(stats)
        except Exception as e:
            logger.error(f"Error getting contradiction stats: {e}")
            return error_response(safe_error_message(e), status=500)
