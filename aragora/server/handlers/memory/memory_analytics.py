"""
Memory analytics endpoint handlers.

Endpoints:
- GET /api/memory/analytics - Get comprehensive memory tier analytics
- GET /api/memory/analytics/tier/{tier} - Get stats for specific tier
- POST /api/memory/analytics/snapshot - Take a manual snapshot
"""

from __future__ import annotations

import logging
from typing import Optional


from ..base import (
    HandlerResult,
    error_response,
    get_clamped_int_param,
    handle_errors,
    json_response,
)
from ..secure import ForbiddenError, SecureHandler, UnauthorizedError
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# RBAC permission for memory analytics endpoints
MEMORY_ANALYTICS_PERMISSION = "memory:analytics"

# Rate limiter for memory analytics endpoints (30 requests per minute - query-heavy)
_memory_analytics_limiter = RateLimiter(requests_per_minute=30)


class MemoryAnalyticsHandler(SecureHandler):
    """Handler for memory analytics endpoints.

    Requires authentication and memory:read permission (RBAC).
    """

    ROUTES = [
        "/api/v1/memory/analytics",
        "/api/v1/memory/analytics/snapshot",
    ]

    def __init__(self, ctx: dict = None):
        """Initialize with context."""
        super().__init__(ctx or {})  # type: ignore[arg-type]
        self._tracker = None

    @property
    def tracker(self):
        """Lazy-load analytics tracker."""
        if self._tracker is None:
            try:
                from aragora.memory.tier_analytics import TierAnalyticsTracker

                db_path = self.ctx.get("analytics_db", "memory_analytics.db")
                self._tracker = TierAnalyticsTracker(db_path=db_path)
            except ImportError as e:
                logger.debug(f"TierAnalyticsTracker not available: {e}")
        return self._tracker

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        if path in self.ROUTES:
            return True
        # Handle tier-specific routes
        if path.startswith("/api/v1/memory/analytics/tier/"):
            return True
        return False

    async def handle(  # type: ignore[override]
        self, path: str, query_params: dict, handler=None
    ) -> Optional[HandlerResult]:
        """Route GET requests with RBAC."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _memory_analytics_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for memory analytics endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and memory:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, MEMORY_ANALYTICS_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required to access memory analytics", 401)
        except ForbiddenError as e:
            logger.warning(f"Memory analytics access denied: {e}")
            return error_response(str(e), 403)

        if path == "/api/v1/memory/analytics":
            days = get_clamped_int_param(query_params, "days", 30, min_val=1, max_val=365)
            return self._get_analytics(days)

        if path.startswith("/api/v1/memory/analytics/tier/"):
            tier_name = path.split("/")[-1]
            days = get_clamped_int_param(query_params, "days", 30, min_val=1, max_val=365)
            return self._get_tier_stats(tier_name, days)

        return None

    async def handle_post(  # type: ignore[override]
        self, path: str, body: dict, handler=None
    ) -> Optional[HandlerResult]:
        """Handle POST requests with RBAC."""
        # RBAC: Require authentication and memory:read permission for POST too
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, MEMORY_ANALYTICS_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"Memory analytics POST access denied: {e}")
            return error_response(str(e), 403)

        if path == "/api/v1/memory/analytics/snapshot":
            return self._take_snapshot()
        return None

    @handle_errors("memory analytics")
    def _get_analytics(self, days: int) -> HandlerResult:
        """Get comprehensive memory analytics.

        Returns:
            Analytics including tier stats, promotion effectiveness,
            learning velocity, and recommendations
        """
        if not self.tracker:
            return error_response("Memory analytics module not available", 503)

        analytics = self.tracker.get_analytics(days=days)

        return json_response(analytics.to_dict())

    @handle_errors("tier stats")
    def _get_tier_stats(self, tier_name: str, days: int) -> HandlerResult:
        """Get stats for a specific memory tier.

        Args:
            tier_name: Name of the tier (fast, medium, slow, glacial)
            days: Number of days to analyze

        Returns:
            TierStats for the specified tier
        """
        if not self.tracker:
            return error_response("Memory analytics module not available", 503)

        try:
            from aragora.memory.tier_manager import MemoryTier

            # Validate tier name
            tier_name_upper = tier_name.upper()
            if tier_name_upper not in [t.name for t in MemoryTier]:
                return error_response(
                    f"Invalid tier: {tier_name}. Valid tiers: {[t.value for t in MemoryTier]}",
                    400,
                )

            tier = MemoryTier[tier_name_upper]
            stats = self.tracker.get_tier_stats(tier, days=days)

            return json_response(stats.to_dict())

        except ImportError:
            return error_response("Memory tier module not available", 503)

    @handle_errors("snapshot")
    def _take_snapshot(self) -> HandlerResult:
        """Take a manual analytics snapshot.

        Returns:
            Confirmation of snapshot
        """
        if not self.tracker:
            return error_response("Memory analytics module not available", 503)

        self.tracker.take_snapshot()

        return json_response(
            {
                "status": "success",
                "message": "Snapshot recorded for all tiers",
            }
        )
