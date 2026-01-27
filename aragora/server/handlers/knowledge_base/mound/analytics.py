"""
Analytics operations mixin for Knowledge Mound handler.

Provides HTTP endpoints for knowledge analytics:
- GET /api/knowledge/mound/analytics/coverage - Domain coverage analysis
- GET /api/knowledge/mound/analytics/usage - Usage pattern analysis
- POST /api/knowledge/mound/analytics/usage/record - Record usage event
- GET /api/knowledge/mound/analytics/quality - Quality snapshot
- GET /api/knowledge/mound/analytics/quality/trend - Quality trend over time
- GET /api/knowledge/mound/analytics/stats - Analytics statistics

Phase A2 - Knowledge Analytics Dashboard
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


class AnalyticsOperationsMixin:
    """Mixin providing analytics API endpoints."""

    ctx: Dict[str, Any]

    def _get_mound(self) -> Optional["KnowledgeMound"]:
        """Get the knowledge mound instance."""
        raise NotImplementedError("Subclass must implement _get_mound")

    @require_permission("analytics:read")
    @rate_limit(requests_per_minute=20)
    async def analyze_coverage(
        self,
        workspace_id: str,
        stale_threshold_days: int = 90,
    ) -> HandlerResult:
        """
        Analyze domain coverage for a workspace.

        GET /api/knowledge/mound/analytics/coverage?workspace_id=...&stale_threshold_days=90

        Args:
            workspace_id: Workspace to analyze
            stale_threshold_days: Days before item is considered stale

        Returns:
            CoverageReport with domain coverage metrics
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            report = await mound.analyze_coverage(
                workspace_id=workspace_id,
                stale_threshold_days=stale_threshold_days,
            )

            return json_response(report.to_dict())
        except Exception as e:
            logger.error(f"Error analyzing coverage: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("analytics:read")
    @rate_limit(requests_per_minute=20)
    async def analyze_usage(
        self,
        workspace_id: str,
        days: int = 30,
    ) -> HandlerResult:
        """
        Analyze usage patterns for a workspace.

        GET /api/knowledge/mound/analytics/usage?workspace_id=...&days=30

        Args:
            workspace_id: Workspace to analyze
            days: Number of days to look back

        Returns:
            UsageReport with usage patterns
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            report = await mound.analyze_usage(
                workspace_id=workspace_id,
                days=days,
            )

            return json_response(report.to_dict())
        except Exception as e:
            logger.error(f"Error analyzing usage: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("analytics:read")
    @rate_limit(requests_per_minute=100)
    async def record_usage_event(
        self,
        event_type: str,
        item_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> HandlerResult:
        """
        Record a usage event.

        POST /api/knowledge/mound/analytics/usage/record
        {
            "event_type": "query",  // query, view, cite, share, export
            "item_id": "...",
            "user_id": "...",
            "workspace_id": "...",
            "query": "search query..."
        }

        Args:
            event_type: Type of usage event
            item_id: Item involved (optional)
            user_id: User who triggered (optional)
            workspace_id: Workspace context (optional)
            query: Query string if applicable (optional)

        Returns:
            Created usage event
        """
        from aragora.knowledge.mound.ops.analytics import UsageEventType

        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not event_type:
            return error_response("event_type is required", status=400)

        try:
            event_type_enum = UsageEventType(event_type)
        except ValueError:
            valid_types = [t.value for t in UsageEventType]
            return error_response(f"Invalid event_type. Valid types: {valid_types}", status=400)

        try:
            event = await mound.record_usage_event(
                event_type=event_type_enum,
                item_id=item_id,
                user_id=user_id,
                workspace_id=workspace_id,
                query=query,
            )

            return json_response(
                {
                    "success": True,
                    "event_id": event.id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"Error recording usage event: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("analytics:read")
    @rate_limit(requests_per_minute=30)
    async def capture_quality_snapshot(
        self,
        workspace_id: str,
    ) -> HandlerResult:
        """
        Capture current quality metrics snapshot.

        POST /api/knowledge/mound/analytics/quality/snapshot
        {
            "workspace_id": "..."
        }

        Args:
            workspace_id: Workspace to snapshot

        Returns:
            QualitySnapshot with current metrics
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            snapshot = await mound.capture_quality_snapshot(
                workspace_id=workspace_id,
            )

            return json_response(snapshot.to_dict())
        except Exception as e:
            logger.error(f"Error capturing quality snapshot: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("analytics:read")
    @rate_limit(requests_per_minute=30)
    async def get_quality_trend(
        self,
        workspace_id: str,
        days: int = 30,
    ) -> HandlerResult:
        """
        Get quality trend over time.

        GET /api/knowledge/mound/analytics/quality/trend?workspace_id=...&days=30

        Args:
            workspace_id: Workspace to analyze
            days: Number of days to look back

        Returns:
            QualityTrend with historical snapshots
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        if not workspace_id:
            return error_response("workspace_id is required", status=400)

        try:
            trend = await mound.get_quality_trend(
                workspace_id=workspace_id,
                days=days,
            )

            return json_response(trend.to_dict())
        except Exception as e:
            logger.error(f"Error getting quality trend: {e}")
            return error_response(safe_error_message(e), status=500)

    @require_permission("analytics:read")
    @rate_limit(requests_per_minute=60)
    async def get_analytics_stats(self) -> HandlerResult:
        """
        Get analytics statistics.

        GET /api/knowledge/mound/analytics/stats

        Returns:
            Analytics statistics
        """
        mound = self._get_mound()
        if not mound:
            return error_response("Knowledge mound not available", status=503)

        try:
            stats = mound.get_analytics_stats()
            return json_response(stats)
        except Exception as e:
            logger.error(f"Error getting analytics stats: {e}")
            return error_response(safe_error_message(e), status=500)
