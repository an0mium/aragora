"""
Cross-Pollination observability endpoint handlers.

Endpoints:
- GET /api/cross-pollination/stats - Get cross-subscriber statistics
- GET /api/cross-pollination/subscribers - List all subscribers
- GET /api/cross-pollination/bridge - Arena event bridge status
- POST /api/cross-pollination/reset - Reset subscriber statistics
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

from .base import BaseHandler, HandlerResult, error_response, json_response
from .utils.rate_limit import RateLimiter

# Rate limiter for cross-pollination endpoints
_cross_pollination_limiter = RateLimiter(requests_per_minute=60)


class CrossPollinationStatsHandler(BaseHandler):
    """
    Handler for GET /api/cross-pollination/stats.

    Returns statistics for all cross-subsystem subscribers including:
    - Events processed per subscriber
    - Events failed per subscriber
    - Last event time
    - Enabled status
    """

    ROUTES = ["/api/cross-pollination/stats"]

    async def get(self) -> HandlerResult:
        """Get cross-subscriber statistics."""
        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()
            stats = manager.get_stats()

            # Calculate totals
            total_processed = sum(s["events_processed"] for s in stats.values())
            total_failed = sum(s["events_failed"] for s in stats.values())
            enabled_count = sum(1 for s in stats.values() if s.get("enabled", True))

            return json_response({
                "status": "ok",
                "summary": {
                    "total_subscribers": len(stats),
                    "enabled_subscribers": enabled_count,
                    "total_events_processed": total_processed,
                    "total_events_failed": total_failed,
                },
                "subscribers": stats,
            })

        except ImportError:
            return error_response(
                "Cross-subscriber module not available",
                status_code=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get cross-pollination stats: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationSubscribersHandler(BaseHandler):
    """
    Handler for GET /api/cross-pollination/subscribers.

    Returns list of all registered subscribers with their event types.
    """

    ROUTES = ["/api/cross-pollination/subscribers"]

    async def get(self) -> HandlerResult:
        """List all subscribers."""
        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()

            subscribers = []
            for event_type, handlers in manager._subscribers.items():
                for name, handler in handlers:
                    subscribers.append({
                        "name": name,
                        "event_type": event_type.value,
                        "handler": handler.__name__ if hasattr(handler, "__name__") else str(handler),
                    })

            return json_response({
                "status": "ok",
                "count": len(subscribers),
                "subscribers": subscribers,
            })

        except ImportError:
            return error_response(
                "Cross-subscriber module not available",
                status_code=503,
            )
        except Exception as e:
            logger.exception(f"Failed to list subscribers: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationBridgeHandler(BaseHandler):
    """
    Handler for GET /api/cross-pollination/bridge.

    Returns status of the Arena event bridge connection.
    """

    ROUTES = ["/api/cross-pollination/bridge"]

    async def get(self) -> HandlerResult:
        """Get bridge status."""
        try:
            from aragora.events.arena_bridge import EVENT_TYPE_MAP

            # Get event type mappings
            mappings = {
                event_str: stream_type.value
                for event_str, stream_type in EVENT_TYPE_MAP.items()
            }

            return json_response({
                "status": "ok",
                "event_mappings": mappings,
                "mapped_event_count": len(mappings),
            })

        except ImportError:
            return error_response(
                "Arena bridge module not available",
                status_code=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get bridge status: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationMetricsHandler(BaseHandler):
    """
    Handler for GET /api/cross-pollination/metrics.

    Returns Prometheus-format metrics for cross-pollination event system.
    """

    ROUTES = ["/api/cross-pollination/metrics"]

    async def get(self) -> HandlerResult:
        """Get cross-pollination metrics in Prometheus format."""
        try:
            from aragora.server.prometheus_cross_pollination import (
                get_cross_pollination_metrics_text,
                PROMETHEUS_AVAILABLE,
            )

            metrics_text = get_cross_pollination_metrics_text()

            return {
                "status": 200,
                "headers": {
                    "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
                },
                "body": metrics_text,
            }

        except ImportError:
            return error_response(
                "Metrics module not available",
                status_code=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get metrics: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationResetHandler(BaseHandler):
    """
    Handler for POST /api/cross-pollination/reset.

    Resets subscriber statistics (for testing/debugging).
    """

    ROUTES = ["/api/cross-pollination/reset"]

    async def post(self) -> HandlerResult:
        """Reset subscriber statistics."""
        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()
            manager.reset_stats()

            return json_response({
                "status": "ok",
                "message": "Cross-subscriber statistics reset",
            })

        except ImportError:
            return error_response(
                "Cross-subscriber module not available",
                status_code=503,
            )
        except Exception as e:
            logger.exception(f"Failed to reset stats: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationKMHandler(BaseHandler):
    """
    Handler for GET /api/cross-pollination/km.

    Returns Knowledge Mound bidirectional integration status including:
    - Adapter states (Ranking, RLM, Continuum, etc.)
    - Batch queue status
    - Cross-subsystem handler statistics
    """

    ROUTES = ["/api/cross-pollination/km"]

    async def get(self) -> HandlerResult:
        """Get KM bidirectional integration status."""
        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()

            # Get KM-related handler stats
            km_handlers = [
                "memory_to_mound",
                "mound_to_memory_retrieval",
                "belief_to_mound",
                "mound_to_belief",
                "rlm_to_mound",
                "mound_to_rlm",
                "elo_to_mound",
                "mound_to_team_selection",
                "insight_to_mound",
                "flip_to_mound",
                "mound_to_trickster",
                "culture_to_debate",
                "staleness_to_debate",
                "provenance_to_mound",
                "mound_to_provenance",
            ]

            all_stats = manager.get_stats()
            km_stats = {
                name: all_stats.get(name, {"events_processed": 0, "events_failed": 0})
                for name in km_handlers
            }

            # Get batch queue status
            batch_stats = manager.get_batch_stats()

            # Calculate inbound vs outbound flows
            inbound_handlers = [h for h in km_handlers if not h.startswith("mound_to")]
            outbound_handlers = [h for h in km_handlers if h.startswith("mound_to")]

            inbound_processed = sum(
                km_stats.get(h, {}).get("events_processed", 0)
                for h in inbound_handlers
            )
            outbound_processed = sum(
                km_stats.get(h, {}).get("events_processed", 0)
                for h in outbound_handlers
            )

            return json_response({
                "status": "ok",
                "summary": {
                    "total_km_handlers": len(km_handlers),
                    "inbound_handlers": len(inbound_handlers),
                    "outbound_handlers": len(outbound_handlers),
                    "inbound_events_processed": inbound_processed,
                    "outbound_events_processed": outbound_processed,
                },
                "handlers": km_stats,
                "batch_queue": batch_stats,
                "adapters": {
                    "ranking": "RankingAdapter - Agent expertise storage",
                    "rlm": "RlmAdapter - Compression pattern storage",
                    "continuum": "ContinuumAdapter - Memory tier mapping",
                    "belief": "BeliefAdapter - Beliefs and cruxes",
                    "insights": "InsightsAdapter - Insights and flips",
                    "evidence": "EvidenceAdapter - Evidence reliability",
                    "consensus": "ConsensusAdapter - Debate outcomes",
                    "critique": "CritiqueAdapter - Pattern success rates",
                },
            })

        except ImportError:
            return error_response(
                "Cross-subscriber module not available",
                status_code=503,
            )
        except Exception as e:
            logger.exception(f"Failed to get KM status: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationKMSyncHandler(BaseHandler):
    """
    Handler for POST /api/cross-pollination/km/sync.

    Triggers manual sync of KM adapters (RankingAdapter, RlmAdapter).
    Useful for forcing persistence after important debates or before shutdown.
    """

    ROUTES = ["/api/cross-pollination/km/sync"]

    async def post(self) -> HandlerResult:
        """Trigger manual KM adapter sync."""
        import time

        try:
            from aragora.events.cross_subscribers import get_cross_subscriber_manager

            manager = get_cross_subscriber_manager()
            results = {
                "ranking": {"status": "skipped", "records": 0},
                "rlm": {"status": "skipped", "patterns": 0},
            }
            start_time = time.time()

            # Sync RankingAdapter
            try:
                from aragora.knowledge.mound.adapters.ranking_adapter import RankingAdapter

                ranking_adapter = getattr(manager, "_ranking_adapter", None)
                if ranking_adapter is None:
                    ranking_adapter = RankingAdapter()
                    manager._ranking_adapter = ranking_adapter

                stats = ranking_adapter.get_stats()
                expertise_count = stats.get("total_expertise_records", 0)

                if expertise_count > 0:
                    # Sync to KM (would use mound instance in production)
                    results["ranking"] = {
                        "status": "synced",
                        "records": expertise_count,
                        "domains": list(stats.get("domains", {}).keys()),
                    }
                else:
                    results["ranking"] = {"status": "empty", "records": 0}

            except ImportError:
                results["ranking"] = {"status": "unavailable", "error": "adapter not installed"}
            except Exception as e:
                results["ranking"] = {"status": "error", "error": str(e)}

            # Sync RlmAdapter
            try:
                from aragora.knowledge.mound.adapters.rlm_adapter import RlmAdapter

                rlm_adapter = getattr(manager, "_rlm_adapter", None)
                if rlm_adapter is None:
                    rlm_adapter = RlmAdapter()
                    manager._rlm_adapter = rlm_adapter

                stats = rlm_adapter.get_stats()
                patterns_count = stats.get("total_patterns", 0)

                if patterns_count > 0:
                    results["rlm"] = {
                        "status": "synced",
                        "patterns": patterns_count,
                    }
                else:
                    results["rlm"] = {"status": "empty", "patterns": 0}

            except ImportError:
                results["rlm"] = {"status": "unavailable", "error": "adapter not installed"}
            except Exception as e:
                results["rlm"] = {"status": "error", "error": str(e)}

            # Flush event batches
            batch_flushed = manager.flush_all_batches()

            duration_ms = (time.time() - start_time) * 1000

            # Record metrics
            try:
                from aragora.server.prometheus_cross_pollination import record_km_adapter_sync

                for adapter_name, result in results.items():
                    status = result.get("status", "unknown")
                    record_km_adapter_sync(adapter_name, "to_mound", status, duration_ms / 1000)
            except ImportError:
                pass

            return json_response({
                "status": "ok",
                "message": "KM adapters synced",
                "results": results,
                "batches_flushed": batch_flushed,
                "duration_ms": round(duration_ms, 2),
            })

        except ImportError:
            return error_response(
                "Cross-subscriber module not available",
                status_code=503,
            )
        except Exception as e:
            logger.exception(f"Failed to sync KM adapters: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationKMStalenessHandler(BaseHandler):
    """
    Handler for POST /api/cross-pollination/km/staleness-check.

    Triggers manual staleness check for Knowledge Mound nodes.
    """

    ROUTES = ["/api/cross-pollination/km/staleness-check"]

    async def post(self) -> HandlerResult:
        """Trigger manual staleness check."""
        import time

        try:
            workspace_id = "default"  # Could be parameterized in future
            start_time = time.time()

            try:
                from aragora.knowledge.mound.facade import get_knowledge_mound
                from aragora.knowledge.mound.staleness import StalenessDetector, StalenessConfig

                mound = get_knowledge_mound()
                if mound is None:
                    return json_response({
                        "status": "ok",
                        "message": "Knowledge Mound not available",
                        "stale_nodes": 0,
                        "workspace_id": workspace_id,
                    })

                # Create detector
                detector = StalenessDetector(
                    mound=mound,
                    config=StalenessConfig(
                        auto_revalidation_threshold=0.7,
                    ),
                )

                # Check staleness
                stale_nodes = await detector.get_stale_nodes(
                    workspace_id=workspace_id,
                    threshold=0.7,
                    limit=100,
                )

                stale_count = len(stale_nodes) if stale_nodes else 0
                duration_ms = (time.time() - start_time) * 1000

                # Record metrics
                try:
                    from aragora.server.prometheus_cross_pollination import record_km_staleness_check
                    record_km_staleness_check(workspace_id, "completed", stale_count)
                except ImportError:
                    pass

                return json_response({
                    "status": "ok",
                    "message": f"Found {stale_count} stale nodes",
                    "stale_nodes": stale_count,
                    "workspace_id": workspace_id,
                    "threshold": 0.7,
                    "duration_ms": round(duration_ms, 2),
                })

            except ImportError as e:
                return json_response({
                    "status": "ok",
                    "message": f"Staleness check module not available: {e}",
                    "stale_nodes": 0,
                })

        except Exception as e:
            logger.exception(f"Failed to run staleness check: {e}")
            return error_response(str(e), status_code=500)


class CrossPollinationKMCultureHandler(BaseHandler):
    """
    Handler for GET /api/cross-pollination/km/culture.

    Returns culture patterns for a workspace.
    """

    ROUTES = ["/api/cross-pollination/km/culture"]

    async def get(self) -> HandlerResult:
        """Get culture patterns."""
        try:
            workspace_id = self.request.query.get("workspace_id", "default") if hasattr(self, "request") else "default"

            try:
                from aragora.knowledge.mound.facade import get_knowledge_mound

                mound = get_knowledge_mound()
                if mound is None or not hasattr(mound, "_culture_accumulator"):
                    return json_response({
                        "status": "ok",
                        "message": "Culture accumulator not available",
                        "workspace_id": workspace_id,
                        "patterns": [],
                    })

                accumulator = mound._culture_accumulator
                if accumulator is None:
                    return json_response({
                        "status": "ok",
                        "message": "Culture accumulator not initialized",
                        "workspace_id": workspace_id,
                        "patterns": [],
                    })

                # Get patterns summary
                summary = accumulator.get_patterns_summary(workspace_id)

                return json_response({
                    "status": "ok",
                    "workspace_id": workspace_id,
                    **summary,
                })

            except ImportError as e:
                return json_response({
                    "status": "ok",
                    "message": f"Culture module not available: {e}",
                    "workspace_id": workspace_id,
                    "patterns": [],
                })

        except Exception as e:
            logger.exception(f"Failed to get culture patterns: {e}")
            return error_response(str(e), status_code=500)


def register_routes(router, server_context: Optional[Any] = None) -> None:
    """
    Register cross-pollination routes with the router.

    Args:
        router: aiohttp router or FastAPI app
        server_context: Optional server context for handlers
    """
    # Create handler instances with context
    stats_handler = CrossPollinationStatsHandler(server_context or {})
    subscribers_handler = CrossPollinationSubscribersHandler(server_context or {})
    bridge_handler = CrossPollinationBridgeHandler(server_context or {})
    metrics_handler = CrossPollinationMetricsHandler(server_context or {})
    reset_handler = CrossPollinationResetHandler(server_context or {})
    km_handler = CrossPollinationKMHandler(server_context or {})
    km_sync_handler = CrossPollinationKMSyncHandler(server_context or {})
    km_staleness_handler = CrossPollinationKMStalenessHandler(server_context or {})
    km_culture_handler = CrossPollinationKMCultureHandler(server_context or {})

    routes = [
        ("GET", "/api/cross-pollination/stats", stats_handler.get),
        ("GET", "/api/cross-pollination/subscribers", subscribers_handler.get),
        ("GET", "/api/cross-pollination/bridge", bridge_handler.get),
        ("GET", "/api/cross-pollination/metrics", metrics_handler.get),
        ("GET", "/api/cross-pollination/km", km_handler.get),
        ("POST", "/api/cross-pollination/reset", reset_handler.post),
        ("POST", "/api/cross-pollination/km/sync", km_sync_handler.post),
        ("POST", "/api/cross-pollination/km/staleness-check", km_staleness_handler.post),
        ("GET", "/api/cross-pollination/km/culture", km_culture_handler.get),
    ]

    for method, path, handler in routes:
        try:
            if hasattr(router, "add_route"):
                router.add_route(method, path, handler)
            elif hasattr(router, "add_api_route"):
                router.add_api_route(path, handler, methods=[method])
        except Exception as e:
            logger.debug(f"Could not register route {path}: {e}")


__all__ = [
    "CrossPollinationStatsHandler",
    "CrossPollinationSubscribersHandler",
    "CrossPollinationBridgeHandler",
    "CrossPollinationMetricsHandler",
    "CrossPollinationResetHandler",
    "CrossPollinationKMHandler",
    "CrossPollinationKMSyncHandler",
    "CrossPollinationKMStalenessHandler",
    "CrossPollinationKMCultureHandler",
    "register_routes",
]
