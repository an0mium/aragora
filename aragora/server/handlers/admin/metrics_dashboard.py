"""
Admin Metrics Dashboard Endpoints.

Provides administrative endpoints for platform metrics and statistics.
All endpoints require admin or owner role with MFA enabled.

Endpoints:
- GET /api/v1/admin/stats - Get system-wide statistics
- GET /api/v1/admin/system/metrics - Get aggregated system metrics
- GET /api/v1/admin/revenue - Get revenue and billing statistics
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from ..base import (
    HandlerResult,
    handle_errors,
    json_response,
)
from ..openapi_decorator import api_endpoint
from aragora.rbac.decorators import require_permission

if TYPE_CHECKING:
    from aragora.auth.context import AuthorizationContext
    from aragora.auth.store import UserStore

logger = logging.getLogger(__name__)


class MetricsDashboardMixin:
    """
    Mixin providing metrics and statistics endpoints for admin dashboard.

    This mixin requires the following attributes from the base class:
    - ctx: dict[str, Any] - Server context
    - _get_user_store() -> UserStore | None
    - _require_admin(handler) -> tuple[AuthContext | None, HandlerResult | None]
    - _check_rbac_permission(auth_ctx, permission, resource_id=None) -> HandlerResult | None
    """

    # Type stubs for methods expected from host class (BaseHandler)
    ctx: dict[str, Any]
    _require_admin: Callable[[Any], tuple[AuthorizationContext | None, HandlerResult | None]]
    _check_rbac_permission: Callable[..., HandlerResult | None]
    _get_user_store: Callable[[], UserStore | None]

    @api_endpoint(
        method="GET",
        path="/api/v1/admin/stats",
        summary="Get system-wide statistics",
        tags=["Admin"],
        operation_id="get_admin_stats",
        responses={
            "200": {
                "description": "System-wide statistics",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"stats": {"type": "object"}},
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @handle_errors("get admin stats")
    def _get_stats(self, handler: Any) -> HandlerResult:
        """Get system-wide statistics.
        Requires admin:stats:read permission.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission
        perm_err = self._check_rbac_permission(auth_ctx, "admin.stats.read")
        if perm_err:
            return perm_err

        user_store = self._get_user_store()
        stats = user_store.get_admin_stats()

        return json_response({"stats": stats})

    @api_endpoint(
        method="GET",
        path="/api/v1/admin/system/metrics",
        summary="Get aggregated system metrics",
        tags=["Admin"],
        responses={
            "200": {
                "description": "Aggregated system metrics from all sources",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"metrics": {"type": "object"}},
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @handle_errors("get system metrics")
    def _get_system_metrics(self, handler: Any) -> HandlerResult:
        """Get aggregated system metrics from various sources.
        Requires admin:metrics:read permission.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission
        perm_err = self._check_rbac_permission(auth_ctx, "admin.metrics.read")
        if perm_err:
            return perm_err

        metrics: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Get user store stats
        user_store = self._get_user_store()
        if user_store:
            metrics["users"] = user_store.get_admin_stats()

        # Get debate storage stats if available
        debate_storage = self.ctx.get("debate_storage")
        if debate_storage and hasattr(debate_storage, "get_statistics"):
            try:
                metrics["debates"] = debate_storage.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get debate stats: {e}")
                metrics["debates"] = {"error": "unavailable"}

        # Get circuit breaker stats if available
        try:
            from aragora.resilience import get_circuit_breaker_status

            metrics["circuit_breakers"] = get_circuit_breaker_status()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to get circuit breaker stats: {e}")

        # Get cache stats if available
        try:
            from aragora.server.handlers.admin.cache import get_cache_stats

            metrics["cache"] = get_cache_stats()
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")

        # Get rate limit stats if available
        try:
            from aragora.server.middleware.rate_limit import get_rate_limiter

            limiter = get_rate_limiter()
            if limiter and hasattr(limiter, "get_stats"):
                metrics["rate_limits"] = limiter.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get rate limit stats: {e}")

        return json_response({"metrics": metrics})

    @api_endpoint(
        method="GET",
        path="/api/v1/admin/revenue",
        summary="Get revenue and billing statistics",
        tags=["Admin"],
        responses={
            "200": {
                "description": "Revenue statistics including MRR and ARR",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"revenue": {"type": "object"}},
                        }
                    }
                },
            },
            "401": {"description": "Unauthorized"},
            "403": {"description": "Forbidden - requires admin role and MFA"},
        },
    )
    @require_permission("admin:revenue:read")
    @handle_errors("get revenue stats")
    def _get_revenue_stats(self, handler: Any) -> HandlerResult:
        """Get revenue and billing statistics.
        Requires admin:revenue:read permission.
        """
        auth_ctx, err = self._require_admin(handler)
        if err:
            return err

        # Check RBAC permission
        perm_err = self._check_rbac_permission(auth_ctx, "admin.revenue.read")
        if perm_err:
            return perm_err

        user_store = self._get_user_store()

        # Get tier distribution from stats
        stats = user_store.get_admin_stats()
        tier_distribution = stats.get("tier_distribution", {})

        # Calculate monthly recurring revenue (MRR) based on tier counts
        from aragora.billing.models import TIER_LIMITS

        mrr_cents = 0
        tier_revenue = {}
        for tier_name, count in tier_distribution.items():
            tier_limits = TIER_LIMITS.get(tier_name)
            if tier_limits:
                tier_mrr = tier_limits.price_monthly_cents * count
                tier_revenue[tier_name] = {
                    "count": count,
                    "price_cents": tier_limits.price_monthly_cents,
                    "mrr_cents": tier_mrr,
                }
                mrr_cents += tier_mrr

        return json_response(
            {
                "revenue": {
                    "mrr_cents": mrr_cents,
                    "mrr_dollars": mrr_cents / 100,
                    "arr_dollars": (mrr_cents * 12) / 100,
                    "tier_breakdown": tier_revenue,
                    "total_organizations": stats.get("total_organizations", 0),
                    "paying_organizations": sum(
                        count for tier, count in tier_distribution.items() if tier != "free"
                    ),
                }
            }
        )


__all__ = ["MetricsDashboardMixin"]
