"""
HTTP Handler for Knowledge Mound Analytics.

Provides endpoints for analytics data:
- GET /api/knowledge/mound/stats - Get mound statistics
- GET /api/knowledge/sharing/stats - Get sharing statistics
- GET /api/knowledge/federation/stats - Get federation statistics
"""

import logging
from typing import Any, Optional

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

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

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return (
            path.startswith("/api/knowledge/mound/stats")
            or path.startswith("/api/knowledge/sharing/stats")
            or path.startswith("/api/knowledge/federation/stats")
            or path.startswith("/api/knowledge/analytics")
        )

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _analytics_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        # Optional authentication (some stats may be public)
        user_id = None
        try:
            user, err = self.require_auth_or_error(handler)
            if not err and user:
                user_id = user.get("sub") or user.get("user_id") or user.get("id")
        except Exception:
            # Auth not required for analytics
            pass

        workspace_id = query_params.get("workspace_id")

        if path == "/api/knowledge/mound/stats":
            return self._get_mound_stats(workspace_id)

        if path == "/api/knowledge/sharing/stats":
            return self._get_sharing_stats(workspace_id, user_id)

        if path == "/api/knowledge/federation/stats":
            return self._get_federation_stats(workspace_id)

        if path == "/api/knowledge/analytics/summary":
            return self._get_summary(workspace_id, user_id)

        return None

    def _get_mound_stats(self, workspace_id: Optional[str]) -> HandlerResult:
        """Get Knowledge Mound statistics."""
        try:
            # Try to get stats from the knowledge mound
            try:
                import asyncio
                from aragora.knowledge.mound import get_knowledge_mound

                async def fetch_stats():
                    mound = await get_knowledge_mound(workspace_id or "default")
                    return await mound.get_stats(workspace_id)

                # Run async in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    stats = loop.run_until_complete(fetch_stats())

                    return json_response({
                        "total_nodes": stats.total_nodes,
                        "nodes_by_type": stats.nodes_by_type,
                        "nodes_by_tier": stats.nodes_by_tier,
                        "nodes_by_validation": stats.nodes_by_validation,
                        "total_relationships": stats.total_relationships,
                        "relationships_by_type": stats.relationships_by_type,
                        "average_confidence": stats.average_confidence,
                        "stale_nodes_count": stats.stale_nodes_count,
                        "workspace_id": workspace_id,
                    })
                finally:
                    loop.close()

            except ImportError:
                # Knowledge mound not available, return mock data
                return json_response({
                    "total_nodes": 0,
                    "nodes_by_type": {},
                    "nodes_by_tier": {},
                    "nodes_by_validation": {},
                    "total_relationships": 0,
                    "relationships_by_type": {},
                    "average_confidence": 0.0,
                    "stale_nodes_count": 0,
                    "workspace_id": workspace_id,
                })

        except Exception as e:
            logger.error(f"Failed to get mound stats: {e}")
            return error_response("Failed to get mound stats", 500)

    def _get_sharing_stats(
        self,
        workspace_id: Optional[str],
        user_id: Optional[str],
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
                    shared_with_me = len([
                        n for n in notifications
                        if n.notification_type.value == "item_shared"
                    ])

                return json_response({
                    "total_shared_items": total_shared,
                    "items_shared_with_me": shared_with_me,
                    "items_shared_by_me": shared_by_me,
                    "active_grants": active_grants,
                    "expired_grants": expired_grants,
                    "workspace_id": workspace_id,
                })

            except ImportError:
                return json_response({
                    "total_shared_items": 0,
                    "items_shared_with_me": 0,
                    "items_shared_by_me": 0,
                    "active_grants": 0,
                    "expired_grants": 0,
                    "workspace_id": workspace_id,
                })

        except Exception as e:
            logger.error(f"Failed to get sharing stats: {e}")
            return error_response("Failed to get sharing stats", 500)

    def _get_federation_stats(self, workspace_id: Optional[str]) -> HandlerResult:
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
                today_runs = [
                    r for r in history
                    if r.started_at.date() == today
                ]

                items_pushed_today = sum(r.items_pushed for r in today_runs)
                items_pulled_today = sum(r.items_pulled for r in today_runs)

                last_sync = history[0] if history else None

                return json_response({
                    "registered_regions": len(scheduler.list_schedules()),
                    "active_schedules": stats["schedules"]["active"],
                    "total_syncs": stats["runs"]["total"],
                    "items_pushed_today": items_pushed_today,
                    "items_pulled_today": items_pulled_today,
                    "last_sync_at": last_sync.started_at.isoformat() if last_sync else None,
                    "success_rate": stats["recent"]["success_rate"],
                    "workspace_id": workspace_id,
                })

            except ImportError:
                return json_response({
                    "registered_regions": 0,
                    "active_schedules": 0,
                    "total_syncs": 0,
                    "items_pushed_today": 0,
                    "items_pulled_today": 0,
                    "last_sync_at": None,
                    "success_rate": 0,
                    "workspace_id": workspace_id,
                })

        except Exception as e:
            logger.error(f"Failed to get federation stats: {e}")
            return error_response("Failed to get federation stats", 500)

    def _get_summary(
        self,
        workspace_id: Optional[str],
        user_id: Optional[str],
    ) -> HandlerResult:
        """Get combined analytics summary."""
        try:
            # Get all stats
            mound_result = self._get_mound_stats(workspace_id)
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

            return json_response({
                "mound": mound_stats,
                "sharing": sharing_stats,
                "federation": federation_stats,
                "workspace_id": workspace_id,
            })

        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return error_response("Failed to get analytics summary", 500)


__all__ = ["AnalyticsHandler"]
