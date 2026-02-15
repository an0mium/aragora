"""
Debate Spectator SSE Handler.

Provides a Server-Sent Events (SSE) endpoint for real-time debate observation:
- GET /api/v1/debates/:id/spectate  - SSE stream of real-time debate events
"""

import asyncio
import json
import logging
import time
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import HandlerResult, json_response

logger = logging.getLogger(__name__)

# Active SSE collectors keyed by debate_id -> set of queues
# Each client gets its own queue; events are fanned out.
_active_collectors: dict[str, set[asyncio.Queue]] = {}


def get_active_collectors() -> dict[str, set[asyncio.Queue]]:
    """Return the active collectors registry (for wiring from event bridge)."""
    return _active_collectors


def push_spectator_event(
    debate_id: str,
    event_type: str,
    agent: str = "",
    details: str = "",
    metric: float | None = None,
    round_number: int | None = None,
) -> int:
    """Push a spectator event to all SSE clients watching a debate.

    Called from the event bridge or spectator stream hook.

    Returns:
        Number of clients the event was pushed to.
    """
    queues = _active_collectors.get(debate_id)
    if not queues:
        return 0

    event = {
        "type": event_type,
        "timestamp": time.time(),
        "agent": agent or None,
        "details": details or None,
        "metric": metric,
        "round": round_number,
    }

    pushed = 0
    dead: list[asyncio.Queue] = []
    for q in queues:
        try:
            q.put_nowait(event)
            pushed += 1
        except asyncio.QueueFull:
            # Drop oldest event and retry
            try:
                q.get_nowait()
                q.put_nowait(event)
                pushed += 1
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                dead.append(q)

    # Clean up dead queues
    for q in dead:
        queues.discard(q)

    return pushed


@require_permission("debates:read")
async def handle_spectate(
    debate_id: str,
    context: AuthorizationContext,
) -> HandlerResult:
    """Return current spectate status for a debate.

    For actual SSE streaming, the route handler calls ``spectate_sse_generator``
    and writes frames directly. This handler serves as a non-streaming fallback
    that reports whether spectating is available.
    """
    n_clients = len(_active_collectors.get(debate_id, set()))
    return json_response(
        {
            "debate_id": debate_id,
            "spectate_available": True,
            "active_viewers": n_clients,
            "sse_url": f"/api/v1/debates/{debate_id}/spectate",
        }
    )


async def spectate_sse_generator(
    debate_id: str,
    *,
    heartbeat_interval: float = 15.0,
    max_queue_size: int = 256,
):
    """Async generator yielding SSE-formatted event strings.

    Each ``yield`` produces a complete SSE frame (``data: ...\\n\\n``).
    The caller is responsible for writing these to the HTTP response.

    Args:
        debate_id: The debate to observe.
        heartbeat_interval: Seconds between keep-alive comments.
        max_queue_size: Max buffered events per client.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)

    # Register
    if debate_id not in _active_collectors:
        _active_collectors[debate_id] = set()
    _active_collectors[debate_id].add(queue)

    try:
        # Initial connection event
        yield _sse_frame("connected", {"debate_id": debate_id})

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
                yield _sse_frame(event.get("type", "event"), event)
            except asyncio.TimeoutError:
                # Send heartbeat comment to keep connection alive
                yield ": heartbeat\n\n"
    finally:
        # Unregister on disconnect
        collectors = _active_collectors.get(debate_id)
        if collectors is not None:
            collectors.discard(queue)
            if not collectors:
                del _active_collectors[debate_id]


def _sse_frame(event_type: str, data: Any) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n"


def register_spectate_routes(router: Any) -> None:
    """Register the spectate SSE route with the server router."""

    async def spectate_endpoint(request: Any) -> Any:
        debate_id = request.path_params.get("debate_id", "")

        # For frameworks that support StreamingResponse (Starlette/FastAPI)
        try:
            from starlette.responses import StreamingResponse

            async def event_stream():
                async for frame in spectate_sse_generator(debate_id):
                    yield frame

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except ImportError:
            # Fallback: return JSON with SSE URL
            return json_response(
                {
                    "debate_id": debate_id,
                    "spectate_available": True,
                    "sse_url": f"/api/v1/debates/{debate_id}/spectate",
                    "message": "Connect via SSE client",
                }
            )

    router.add_route(
        "GET",
        "/api/v1/debates/{debate_id}/spectate",
        spectate_endpoint,
    )


__all__ = [
    "get_active_collectors",
    "handle_spectate",
    "push_spectator_event",
    "register_spectate_routes",
    "spectate_sse_generator",
]
