"""
Dashboard and Metrics Operations Mixin for Knowledge Mound.

Provides HTTP endpoints for KM health monitoring and observability:
- GET /api/knowledge/mound/dashboard/health - Get KM health status
- GET /api/knowledge/mound/dashboard/metrics - Get detailed metrics
- GET /api/knowledge/mound/dashboard/adapters - Get adapter status
- GET /api/knowledge/mound/dashboard/queries - Get federated query stats
- POST /api/knowledge/mound/dashboard/metrics/reset - Reset metrics

These endpoints power the Knowledge Mound dashboard panel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...base import HandlerResult, error_response

if TYPE_CHECKING:
    from aiohttp.web import Request

logger = logging.getLogger(__name__)


class DashboardOperationsMixin:
    """Mixin providing dashboard and metrics endpoints for Knowledge Mound."""

    async def handle_dashboard_health(self, request: "Request") -> HandlerResult:
        """
        GET /api/knowledge/mound/dashboard/health - Get KM health status.

        Returns:
            JSON response with:
            - status: "healthy" | "degraded" | "unhealthy"
            - checks: individual health check results
            - recommendations: suggested actions
            - timestamp: when check was performed
        """
        try:
            from aragora.knowledge.mound.metrics import get_metrics

            metrics = get_metrics()
            health_report = metrics.get_health()

            return HandlerResult(
                success=True,
                data=health_report.to_dict(),
                message="Health check completed",
            )

        except ImportError:
            return error_response(
                "Metrics module not available",
                status=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get health status: {e}")
            return error_response(str(e))

    async def handle_dashboard_metrics(self, request: "Request") -> HandlerResult:
        """
        GET /api/knowledge/mound/dashboard/metrics - Get detailed metrics.

        Returns:
            JSON response with:
            - stats: operation statistics by type
            - health: current health status
            - config: metrics configuration
            - uptime_seconds: time since metrics initialized
        """
        try:
            from aragora.knowledge.mound.metrics import get_metrics

            metrics = get_metrics()

            return HandlerResult(
                success=True,
                data=metrics.to_dict(),
                message="Metrics retrieved",
            )

        except ImportError:
            return error_response(
                "Metrics module not available",
                status=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get metrics: {e}")
            return error_response(str(e))

    async def handle_dashboard_adapters(self, request: "Request") -> HandlerResult:
        """
        GET /api/knowledge/mound/dashboard/adapters - Get adapter status.

        Returns:
            JSON response with:
            - adapters: list of registered adapters with status
            - total: count of registered adapters
            - enabled: count of enabled adapters
        """
        try:
            # Get the BidirectionalCoordinator if available
            mound = self._get_mound()
            if not mound:
                return HandlerResult(
                    success=True,
                    data={
                        "adapters": [],
                        "total": 0,
                        "enabled": 0,
                        "message": "Knowledge Mound not initialized",
                    },
                )

            # Try to get coordinator from the mound or server context
            coordinator = getattr(mound, "_coordinator", None)
            if coordinator is None:
                # Try from server context
                coordinator = self.server_context.get("km_coordinator")

            if coordinator is None:
                return HandlerResult(
                    success=True,
                    data={
                        "adapters": [],
                        "total": 0,
                        "enabled": 0,
                        "message": "No coordinator available",
                    },
                )

            # Get adapter stats from coordinator
            stats = coordinator.get_stats() if hasattr(coordinator, "get_stats") else {}

            adapters_list = []
            for name, info in stats.get("adapters", {}).items():
                adapters_list.append({
                    "name": name,
                    "enabled": info.get("enabled", False),
                    "priority": info.get("priority", 0),
                    "forward_sync_count": info.get("forward_sync_count", 0),
                    "reverse_sync_count": info.get("reverse_sync_count", 0),
                    "last_sync": info.get("last_sync"),
                    "errors": info.get("errors", 0),
                })

            return HandlerResult(
                success=True,
                data={
                    "adapters": adapters_list,
                    "total": stats.get("total_adapters", len(adapters_list)),
                    "enabled": stats.get("enabled_adapters", 0),
                    "last_sync": stats.get("last_full_sync"),
                },
            )

        except Exception as e:
            logger.exception(f"Failed to get adapter status: {e}")
            return error_response(str(e))

    async def handle_dashboard_queries(self, request: "Request") -> HandlerResult:
        """
        GET /api/knowledge/mound/dashboard/queries - Get federated query stats.

        Returns:
            JSON response with:
            - total_queries: total queries executed
            - successful_queries: successful query count
            - success_rate: percentage of successful queries
            - sources: list of query sources with stats
        """
        try:
            from aragora.knowledge.mound.federated_query import FederatedQueryAggregator

            # Try to get aggregator from server context
            aggregator = self.server_context.get("km_aggregator")
            if aggregator is None:
                # Create a temporary one to get empty stats
                aggregator = FederatedQueryAggregator()

            stats = aggregator.get_stats()

            return HandlerResult(
                success=True,
                data=stats,
                message="Query stats retrieved",
            )

        except ImportError:
            return error_response(
                "Federated query module not available",
                status=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get query stats: {e}")
            return error_response(str(e))

    async def handle_dashboard_metrics_reset(self, request: "Request") -> HandlerResult:
        """
        POST /api/knowledge/mound/dashboard/metrics/reset - Reset metrics.

        This clears all collected metrics and restarts tracking.
        Useful for debugging or starting a fresh monitoring session.
        """
        try:
            from aragora.knowledge.mound.metrics import get_metrics

            metrics = get_metrics()
            metrics.reset()

            return HandlerResult(
                success=True,
                data={"message": "Metrics reset successfully"},
            )

        except ImportError:
            return error_response(
                "Metrics module not available",
                status=503,
            )
        except Exception as e:
            logger.exception(f"Failed to reset metrics: {e}")
            return error_response(str(e))

    async def handle_dashboard_batcher_stats(self, request: "Request") -> HandlerResult:
        """
        GET /api/knowledge/mound/dashboard/batcher - Get event batcher stats.

        Returns:
            JSON response with:
            - total_events_queued: events queued for batching
            - total_events_emitted: events emitted to clients
            - total_batches_emitted: number of batches sent
            - average_batch_size: average events per batch
            - pending_events: events waiting to be batched
        """
        try:
            from aragora.knowledge.mound.websocket_bridge import get_km_bridge

            bridge = get_km_bridge()
            if bridge is None:
                return HandlerResult(
                    success=True,
                    data={
                        "running": False,
                        "message": "Event batcher not initialized",
                    },
                )

            stats = bridge.get_stats()

            return HandlerResult(
                success=True,
                data=stats,
            )

        except ImportError:
            return error_response(
                "WebSocket bridge module not available",
                status=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get batcher stats: {e}")
            return error_response(str(e))
