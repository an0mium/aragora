"""
HTTP Handler for Knowledge Mound Analytics.

Provides endpoints for analytics data:
- GET /api/knowledge/mound/stats - Get mound statistics
- GET /api/knowledge/sharing/stats - Get sharing statistics
- GET /api/knowledge/federation/stats - Get federation statistics
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip, rate_limit
from aragora.rbac.decorators import require_permission


# =============================================================================
# Type Definitions for Learning Stats
# =============================================================================


class KnowledgeReuseStats(TypedDict):
    """Statistics about knowledge reuse across debates."""

    total_queries: int
    queries_with_hits: int
    reuse_rate: float


class ValidationStats(TypedDict):
    """Statistics about validation outcomes."""

    total_validations: int
    positive_validations: int
    negative_validations: int
    accuracy_rate: float


class LearningVelocityStats(TypedDict):
    """Statistics about learning velocity."""

    new_items_today: int
    new_items_this_week: int
    items_promoted: int
    items_demoted: int


class CrossDebateUtilityStats(TypedDict):
    """Statistics about cross-debate utility."""

    avg_utility_score: float
    high_utility_items: int
    low_utility_items: int


class AdapterActivityStats(TypedDict):
    """Statistics about adapter activity."""

    forward_syncs_today: int
    reverse_queries_today: int
    semantic_searches_today: int


class LearningStats(TypedDict):
    """Complete learning statistics structure."""

    knowledge_reuse: KnowledgeReuseStats
    validation: ValidationStats
    learning_velocity: LearningVelocityStats
    cross_debate_utility: CrossDebateUtilityStats
    adapter_activity: AdapterActivityStats
    timestamp: str
    workspace_id: str | None


logger = logging.getLogger(__name__)

# =============================================================================
# RBAC Permissions
# =============================================================================

KNOWLEDGE_READ_PERMISSION = "knowledge:read"

# RBAC imports with fallback
try:
    from aragora.rbac.checker import get_permission_checker
    from aragora.rbac.models import AuthorizationContext as RBACContext

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False

# Rate limiter for analytics endpoints
_analytics_limiter = RateLimiter(requests_per_minute=60)


class AnalyticsHandler(BaseHandler):
    """Handler for knowledge analytics endpoints.

    Endpoints:
        GET /api/knowledge/mound/stats - Get mound statistics
        GET /api/knowledge/sharing/stats - Get sharing statistics
        GET /api/knowledge/federation/stats - Get federation statistics
        GET /api/knowledge/analytics/summary - Get combined summary
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return (
            path.startswith("/api/v1/knowledge/mound/stats")
            or path.startswith("/api/v1/knowledge/sharing/stats")
            or path.startswith("/api/v1/knowledge/federation/stats")
            or path.startswith("/api/v1/knowledge/analytics")
            or path.startswith("/api/v1/knowledge/learning")
        )

    async def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        """Handle GET requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _analytics_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Require authentication for knowledge analytics
        user_id = None
        try:
            user, err = self.require_auth_or_error(handler)
            if err:
                return err
            if user:
                user_id = user.user_id
        except Exception as e:
            logger.warning(f"Authentication failed for knowledge analytics: {e}")
            return error_response("Authentication required", 401)

        # RBAC permission check
        if RBAC_AVAILABLE and user:
            try:
                auth_ctx = RBACContext(
                    user_id=user_id or "anonymous",
                    user_email=user.email,
                    org_id=user.org_id,
                    workspace_id=query_params.get("workspace_id"),
                    roles={user.role} if user else {"member"},
                )
                checker = get_permission_checker()
                decision = checker.check_permission(auth_ctx, KNOWLEDGE_READ_PERMISSION)
                if not decision.allowed:
                    logger.warning(
                        f"Knowledge analytics access denied for {user_id}: {decision.reason}"
                    )
                    return error_response(f"Permission denied: {decision.reason}", 403)
            except Exception as e:
                logger.warning(f"RBAC check failed for knowledge analytics: {e}")
                # Continue without RBAC if it fails (graceful degradation)

        workspace_id = query_params.get("workspace_id")

        if path == "/api/v1/knowledge/mound/stats":
            return await self._get_mound_stats(workspace_id)

        if path == "/api/v1/knowledge/sharing/stats":
            return self._get_sharing_stats(workspace_id, user_id)

        if path == "/api/v1/knowledge/federation/stats":
            return self._get_federation_stats(workspace_id)

        if path == "/api/v1/knowledge/analytics/summary":
            return await self._get_summary(workspace_id, user_id)

        if path == "/api/v1/knowledge/learning/stats":
            return self._get_learning_stats(workspace_id)

        if path == "/api/v1/knowledge/analytics/learning":
            return self._get_learning_stats(workspace_id)

        return None

    @rate_limit(requests_per_minute=60, limiter_name="knowledge_analytics_read")
    async def _get_mound_stats(self, workspace_id: str | None) -> HandlerResult:
        """Get Knowledge Mound statistics."""
        try:
            # Try to get stats from the knowledge mound
            try:
                from aragora.knowledge.mound import get_knowledge_mound

                mound = get_knowledge_mound(workspace_id or "default")
                stats = await mound.get_stats(workspace_id)

                return json_response(
                    {
                        "total_nodes": stats.total_nodes,
                        "nodes_by_type": stats.nodes_by_type,
                        "nodes_by_tier": stats.nodes_by_tier,
                        "nodes_by_validation": stats.nodes_by_validation,
                        "total_relationships": stats.total_relationships,
                        "relationships_by_type": stats.relationships_by_type,
                        "average_confidence": stats.average_confidence,
                        "stale_nodes_count": stats.stale_nodes_count,
                        "workspace_id": workspace_id,
                    }
                )

            except ImportError:
                # Knowledge mound not available, return mock data
                return json_response(
                    {
                        "total_nodes": 0,
                        "nodes_by_type": {},
                        "nodes_by_tier": {},
                        "nodes_by_validation": {},
                        "total_relationships": 0,
                        "relationships_by_type": {},
                        "average_confidence": 0.0,
                        "stale_nodes_count": 0,
                        "workspace_id": workspace_id,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get mound stats: {e}")
            return error_response("Failed to get mound stats", 500)

    @rate_limit(requests_per_minute=60, limiter_name="knowledge_analytics_read")
    def _get_sharing_stats(
        self,
        workspace_id: str | None,
        user_id: str | None,
    ) -> HandlerResult:
        """Get sharing statistics."""
        try:
            # Try to get sharing stats
            try:
                from aragora.knowledge.mound.notifications import get_notification_store

                store = get_notification_store()

                # Calculate sharing stats (simplified)
                # In production, this would query the database
                total_shared = 0
                shared_with_me = 0
                shared_by_me = 0
                active_grants = 0
                expired_grants = 0

                if user_id:
                    notifications = store.get_notifications(user_id, limit=100)
                    shared_with_me = len(
                        [n for n in notifications if n.notification_type.value == "item_shared"]
                    )

                return json_response(
                    {
                        "total_shared_items": total_shared,
                        "items_shared_with_me": shared_with_me,
                        "items_shared_by_me": shared_by_me,
                        "active_grants": active_grants,
                        "expired_grants": expired_grants,
                        "workspace_id": workspace_id,
                    }
                )

            except ImportError:
                return json_response(
                    {
                        "total_shared_items": 0,
                        "items_shared_with_me": 0,
                        "items_shared_by_me": 0,
                        "active_grants": 0,
                        "expired_grants": 0,
                        "workspace_id": workspace_id,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get sharing stats: {e}")
            return error_response("Failed to get sharing stats", 500)

    @rate_limit(requests_per_minute=60, limiter_name="knowledge_analytics_read")
    def _get_federation_stats(self, workspace_id: str | None) -> HandlerResult:
        """Get federation statistics."""
        try:
            # Try to get federation scheduler stats
            try:
                from aragora.knowledge.mound.ops.federation_scheduler import (
                    get_federation_scheduler,
                )

                scheduler = get_federation_scheduler()
                stats = scheduler.get_stats()

                # Calculate today's sync stats
                history = scheduler.get_history(limit=100)
                from datetime import datetime

                today = datetime.now().date()
                today_runs = [r for r in history if r.started_at.date() == today]

                items_pushed_today = sum(r.items_pushed for r in today_runs)
                items_pulled_today = sum(r.items_pulled for r in today_runs)

                last_sync = history[0] if history else None

                return json_response(
                    {
                        "registered_regions": len(scheduler.list_schedules()),
                        "active_schedules": stats["schedules"]["active"],
                        "total_syncs": stats["runs"]["total"],
                        "items_pushed_today": items_pushed_today,
                        "items_pulled_today": items_pulled_today,
                        "last_sync_at": last_sync.started_at.isoformat() if last_sync else None,
                        "success_rate": stats["recent"]["success_rate"],
                        "workspace_id": workspace_id,
                    }
                )

            except ImportError:
                return json_response(
                    {
                        "registered_regions": 0,
                        "active_schedules": 0,
                        "total_syncs": 0,
                        "items_pushed_today": 0,
                        "items_pulled_today": 0,
                        "last_sync_at": None,
                        "success_rate": 0,
                        "workspace_id": workspace_id,
                    }
                )

        except Exception as e:
            logger.error(f"Failed to get federation stats: {e}")
            return error_response("Failed to get federation stats", 500)

    @rate_limit(requests_per_minute=5, limiter_name="knowledge_analytics_expensive")
    async def _get_summary(
        self,
        workspace_id: str | None,
        user_id: str | None,
    ) -> HandlerResult:
        """Get combined analytics summary."""
        try:
            # Get all stats
            mound_result = await self._get_mound_stats(workspace_id)
            sharing_result = self._get_sharing_stats(workspace_id, user_id)
            federation_result = self._get_federation_stats(workspace_id)

            # Parse JSON from results (HandlerResult is a dataclass with .body)
            import json

            def parse_result(result: HandlerResult) -> dict:
                """Parse JSON from HandlerResult body."""
                if result and hasattr(result, "body") and result.status_code == 200:
                    return json.loads(result.body)
                return {}

            mound_stats = parse_result(mound_result)
            sharing_stats = parse_result(sharing_result)
            federation_stats = parse_result(federation_result)

            return json_response(
                {
                    "mound": mound_stats,
                    "sharing": sharing_stats,
                    "federation": federation_stats,
                    "workspace_id": workspace_id,
                }
            )

        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return error_response("Failed to get analytics summary", 500)

    @rate_limit(requests_per_minute=60, limiter_name="knowledge_analytics_read")
    def _get_learning_stats(
        self,
        workspace_id: str | None,
    ) -> HandlerResult:
        """Get cross-debate learning analytics.

        Aggregates metrics that show how effectively the system learns
        across multiple debates, including:
        - Knowledge reuse rate (how often KM data is used in debates)
        - Validation accuracy (how often KM data is validated by outcomes)
        - Learning velocity (rate of knowledge improvement)
        - Cross-debate utility (how useful knowledge is across debates)
        """
        try:
            from datetime import datetime

            # Initialize default stats with proper typed dicts
            knowledge_reuse: KnowledgeReuseStats = {
                "total_queries": 0,
                "queries_with_hits": 0,
                "reuse_rate": 0.0,
            }
            validation: ValidationStats = {
                "total_validations": 0,
                "positive_validations": 0,
                "negative_validations": 0,
                "accuracy_rate": 0.0,
            }
            learning_velocity: LearningVelocityStats = {
                "new_items_today": 0,
                "new_items_this_week": 0,
                "items_promoted": 0,
                "items_demoted": 0,
            }
            cross_debate_utility: CrossDebateUtilityStats = {
                "avg_utility_score": 0.0,
                "high_utility_items": 0,
                "low_utility_items": 0,
            }
            adapter_activity: AdapterActivityStats = {
                "forward_syncs_today": 0,
                "reverse_queries_today": 0,
                "semantic_searches_today": 0,
            }

            # Try to get real stats from adapters
            try:
                from aragora.memory.continuum import get_continuum_memory

                continuum = get_continuum_memory()
                if continuum and hasattr(continuum, "_km_adapter"):
                    adapter = continuum._km_adapter
                    if adapter:
                        stats = adapter.get_stats()

                        # Update with real continuum stats
                        cross_debate_utility["avg_utility_score"] = stats.get(
                            "avg_cross_debate_utility", 0.0
                        )

                        # Count high/low utility items
                        km_validated = stats.get("km_validated_entries", 0)
                        if km_validated > 0:
                            # Estimate based on average
                            avg = stats.get("avg_cross_debate_utility", 0.5)
                            cross_debate_utility["high_utility_items"] = int(km_validated * avg)
                            cross_debate_utility["low_utility_items"] = km_validated - int(
                                km_validated * avg
                            )

            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Failed to get continuum learning stats: {e}")

            # Try to get consensus adapter stats
            # Note: ConsensusMemory doesn't have a singleton getter, so we skip this
            # The consensus memory is typically accessed via storage context

            # Try to get cross-subscriber stats
            try:
                from aragora.events.cross_subscribers import (
                    get_cross_subscriber_manager,
                )

                manager = get_cross_subscriber_manager()
                if manager:
                    cs_stats = manager.get_stats()

                    # Extract validation handler stats
                    handlers = cs_stats.get("handlers", {})
                    validation_stats = handlers.get("km_validation_feedback", {})
                    if validation_stats:
                        validation["total_validations"] = validation_stats.get("call_count", 0)

            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Failed to get cross-subscriber stats: {e}")

            # Calculate derived metrics
            if knowledge_reuse["total_queries"] > 0:
                knowledge_reuse["reuse_rate"] = round(
                    knowledge_reuse["queries_with_hits"] / knowledge_reuse["total_queries"],
                    3,
                )

            if validation["total_validations"] > 0:
                validation["accuracy_rate"] = round(
                    validation["positive_validations"] / validation["total_validations"],
                    3,
                )

            # Assemble the final response
            learning_stats: LearningStats = {
                "knowledge_reuse": knowledge_reuse,
                "validation": validation,
                "learning_velocity": learning_velocity,
                "cross_debate_utility": cross_debate_utility,
                "adapter_activity": adapter_activity,
                "timestamp": datetime.now().isoformat(),
                "workspace_id": workspace_id,
            }

            return json_response(learning_stats)

        except Exception as e:
            logger.error(f"Failed to get learning stats: {e}")
            return error_response("Failed to get learning stats", 500)


__all__ = ["AnalyticsHandler"]
