"""
Debate Spectator SSE Handler.

Provides a Server-Sent Events (SSE) endpoint for real-time debate observation:
- GET /api/v1/debates/:id/spectate  - SSE stream of real-time debate events
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from aragora.rbac.decorators import require_permission
from aragora.rbac.models import AuthorizationContext
from aragora.server.handlers.base import HandlerResult, json_response

logger = logging.getLogger(__name__)

_RECENT_ACTIVITY_WINDOW_SECONDS = 120
_STATUS_ACTIVITY_SCAN_LIMIT = 200

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


def _parse_event_timestamp(timestamp: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if not timestamp:
        return None

    normalized = timestamp.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _debate_bridge_activity(debate_id: str) -> dict[str, Any]:
    """Summarize currently observed live activity for a specific debate."""
    try:
        from aragora.spectate.ws_bridge import get_spectate_bridge
    except ImportError:
        return {
            "bridge_active": False,
            "availability_state": "bridge_inactive",
            "observed_live": False,
            "recent_event_count": 0,
            "last_event_at": None,
            "recent_activity_window_seconds": _RECENT_ACTIVITY_WINDOW_SECONDS,
        }

    bridge = get_spectate_bridge()
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(seconds=_RECENT_ACTIVITY_WINDOW_SECONDS)

    recent_event_count = 0
    last_event_at: str | None = None
    last_event_dt: datetime | None = None

    for event in bridge.get_recent_events(_STATUS_ACTIVITY_SCAN_LIMIT):
        if getattr(event, "debate_id", None) != debate_id:
            continue

        event_timestamp = getattr(event, "timestamp", None)
        event_dt = _parse_event_timestamp(event_timestamp)
        if event_dt and (last_event_dt is None or event_dt > last_event_dt):
            last_event_dt = event_dt
            last_event_at = event_timestamp

        if event_dt and event_dt >= recent_cutoff:
            recent_event_count += 1

    observed_live = recent_event_count > 0
    availability_state = (
        "live" if observed_live else "standby" if bridge.running else "bridge_inactive"
    )

    return {
        "bridge_active": bridge.running,
        "availability_state": availability_state,
        "observed_live": observed_live,
        "recent_event_count": recent_event_count,
        "last_event_at": last_event_at,
        "recent_activity_window_seconds": _RECENT_ACTIVITY_WINDOW_SECONDS,
    }


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
    activity = _debate_bridge_activity(debate_id)
    observed_live = activity["observed_live"] or n_clients > 0
    availability_state = "live" if observed_live else activity["availability_state"]
    return json_response(
        {
            "debate_id": debate_id,
            "spectate_available": True,
            "active_viewers": n_clients,
            **activity,
            "availability_state": availability_state,
            "observed_live": observed_live,
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
