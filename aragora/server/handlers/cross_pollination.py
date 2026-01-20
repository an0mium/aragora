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


class CrossPollinationResetHandler(BaseHandler):
    """
    Handler for POST /api/cross-pollination/reset.

    Resets subscriber statistics (for testing/debugging).
    """

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
    reset_handler = CrossPollinationResetHandler(server_context or {})

    routes = [
        ("GET", "/api/cross-pollination/stats", stats_handler.get),
        ("GET", "/api/cross-pollination/subscribers", subscribers_handler.get),
        ("GET", "/api/cross-pollination/bridge", bridge_handler.get),
        ("POST", "/api/cross-pollination/reset", reset_handler.post),
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
    "CrossPollinationResetHandler",
    "register_routes",
]
