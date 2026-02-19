"""WebSocket/SSE handler for real-time spectate events.

Endpoints:
- GET /api/v1/spectate/recent  - Get recent buffered spectate events
- GET /api/v1/spectate/status  - Get bridge status (active, subscribers, buffer size)
- GET /api/v1/spectate/stream  - SSE endpoint (returns snapshot of recent events)
"""

from __future__ import annotations

__all__ = [
    "SpectateStreamHandler",
]

import logging
from typing import Any

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)

logger = logging.getLogger(__name__)


class SpectateStreamHandler(BaseHandler):
    """Handler for spectate stream endpoints.

    Serves buffered SpectatorStream events over HTTP so that the
    dashboard can poll for live debate/pipeline visualization data.
    """

    ROUTES = [
        "/api/v1/spectate/recent",
        "/api/v1/spectate/status",
        "/api/v1/spectate/stream",
    ]

    @handle_errors
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route GET requests to the appropriate sub-handler."""
        if not path.startswith("/api/v1/spectate"):
            return None

        if path.endswith("/recent"):
            return self._handle_recent(query_params)
        if path.endswith("/status"):
            return self._handle_status()
        if path.endswith("/stream"):
            # SSE stub - returns snapshot for now
            return self._handle_recent(query_params)

        return None

    def _handle_recent(self, query_params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/spectate/recent -- get recent events from the buffer."""
        try:
            from aragora.spectate.ws_bridge import get_spectate_bridge

            bridge = get_spectate_bridge()

            count_str = query_params.get("count", "50") if query_params else "50"
            try:
                count = min(int(count_str), 500)
            except (ValueError, TypeError):
                count = 50

            events = bridge.get_recent_events(count)

            # Optional filtering by debate_id or pipeline_id
            debate_id = query_params.get("debate_id") if query_params else None
            pipeline_id = query_params.get("pipeline_id") if query_params else None

            if debate_id:
                events = [e for e in events if e.debate_id == debate_id]
            if pipeline_id:
                events = [e for e in events if e.pipeline_id == pipeline_id]

            return json_response(
                {
                    "events": [e.to_dict() for e in events],
                    "count": len(events),
                }
            )
        except ImportError:
            return json_response({"events": [], "count": 0})

    def _handle_status(self) -> HandlerResult:
        """GET /api/v1/spectate/status -- bridge status."""
        try:
            from aragora.spectate.ws_bridge import get_spectate_bridge

            bridge = get_spectate_bridge()
            return json_response(
                {
                    "active": bridge.running,
                    "subscribers": bridge.subscriber_count,
                    "buffer_size": bridge.buffer_size,
                }
            )
        except ImportError:
            return json_response({"active": False, "subscribers": 0, "buffer_size": 0})
