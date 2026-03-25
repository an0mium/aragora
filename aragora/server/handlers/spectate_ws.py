"""WebSocket/SSE handler for real-time spectate events.

Endpoints:
- GET /api/v1/spectate/recent  - Get recent buffered spectate events
- GET /api/v1/spectate/status  - Get bridge status (active, subscribers, buffer size)
- GET /api/v1/spectate/stream  - Finite SSE snapshot or JSON preview of recent events
"""

from __future__ import annotations

__all__ = [
    "SpectateStreamHandler",
]

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .base import (
    BaseHandler,
    HandlerResult,
    handle_errors,
    json_response,
)

logger = logging.getLogger(__name__)

_RECENT_ACTIVITY_WINDOW_SECONDS = 120
_STATUS_ACTIVITY_SCAN_LIMIT = 200


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


def _summarize_bridge_activity(events: list[Any], *, bridge_running: bool) -> dict[str, Any]:
    """Summarize recent bridge activity for truthful spectate readiness."""
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(seconds=_RECENT_ACTIVITY_WINDOW_SECONDS)

    last_event_at: str | None = None
    last_event_dt: datetime | None = None
    recent_events: list[Any] = []
    live_debate_summaries: dict[str, dict[str, Any]] = {}

    for event in events:
        event_timestamp = getattr(event, "timestamp", None)
        event_dt = _parse_event_timestamp(event_timestamp)

        if event_dt and (last_event_dt is None or event_dt > last_event_dt):
            last_event_dt = event_dt
            last_event_at = event_timestamp

        if event_dt is None or event_dt < recent_cutoff:
            continue

        recent_events.append(event)
        debate_id = getattr(event, "debate_id", None)
        if not debate_id:
            continue

        summary = live_debate_summaries.setdefault(
            debate_id,
            {
                "debate_id": debate_id,
                "recent_event_count": 0,
                "last_event_at": event_timestamp,
                "_last_event_dt": event_dt,
                "_event_types": set(),
            },
        )
        summary["recent_event_count"] += 1
        if event_dt >= summary["_last_event_dt"]:
            summary["last_event_at"] = event_timestamp
            summary["_last_event_dt"] = event_dt
        summary["_event_types"].add(getattr(event, "event_type", "event"))

    live_debates = [
        {
            "debate_id": debate_id,
            "recent_event_count": summary["recent_event_count"],
            "last_event_at": summary["last_event_at"],
            "event_types": sorted(summary["_event_types"]),
        }
        for debate_id, summary in live_debate_summaries.items()
    ]
    live_debates.sort(
        key=lambda summary: _parse_event_timestamp(summary["last_event_at"])
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    if not bridge_running:
        bridge_state = "inactive"
    elif live_debates:
        bridge_state = "live_debates_available"
    elif recent_events:
        bridge_state = "activity_unattributed"
    else:
        bridge_state = "idle"

    activity_age_seconds = None
    if last_event_dt is not None:
        activity_age_seconds = max((now - last_event_dt).total_seconds(), 0.0)

    return {
        "bridge_state": bridge_state,
        "last_event_at": last_event_at,
        "activity_age_seconds": activity_age_seconds,
        "recent_activity_window_seconds": _RECENT_ACTIVITY_WINDOW_SECONDS,
        "recent_event_count": len(recent_events),
        "live_debate_count": len(live_debates),
        "live_debate_ids": [summary["debate_id"] for summary in live_debates],
        "live_debates": live_debates,
        "unattributed_recent_event_count": len(recent_events)
        - sum(summary["recent_event_count"] for summary in live_debates),
    }


def _redact_live_debate_details(summary: dict[str, Any]) -> dict[str, Any]:
    """Hide debate-specific activity details from unauthenticated callers."""
    redacted = dict(summary)
    if redacted.get("bridge_state") == "live_debates_available":
        redacted["bridge_state"] = "activity_unattributed"
    redacted["live_debate_count"] = 0
    redacted["live_debate_ids"] = []
    redacted["live_debates"] = []
    redacted["unattributed_recent_event_count"] = redacted.get("recent_event_count", 0)
    return redacted


def _sse_frame(event_type: str, data: Any) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data, default=str, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n"


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

    STREAM_MODE = "snapshot"
    STREAM_READINESS = "partial"
    STREAM_JSON_TRANSPORT = "json_preview"
    STREAM_SSE_TRANSPORT = "sse_snapshot"
    STREAM_JSON_MESSAGE = (
        "Buffered spectate events are available as a JSON preview on this endpoint. "
        "Request Accept: text/event-stream or ?format=sse for a finite SSE snapshot; "
        "full live SSE delivery has not shipped yet."
    )
    STREAM_SSE_MESSAGE = (
        "Buffered spectate events are being delivered as a finite SSE snapshot on this "
        "endpoint; full live SSE delivery has not shipped yet."
    )

    @handle_errors("spectate")
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        """Route GET requests to the appropriate sub-handler."""
        if not path.startswith("/api/v1/spectate"):
            return None

        if path.endswith("/recent"):
            return self._handle_recent(query_params)
        if path.endswith("/status"):
            return self._handle_status(handler)
        if path.endswith("/stream"):
            return self._handle_stream(query_params, handler)

        return None

    def _handle_recent(self, query_params: dict[str, Any]) -> HandlerResult:
        """GET /api/v1/spectate/recent -- get recent events from the buffer."""
        events = self._get_recent_events(query_params)
        return json_response(self._recent_payload(events))

    def _handle_stream(self, query_params: dict[str, Any], handler: Any) -> HandlerResult:
        """GET /api/v1/spectate/stream -- finite SSE snapshot or JSON preview."""
        events = self._get_recent_events(query_params)
        if self._wants_sse(query_params, handler):
            metadata = self._stream_metadata(
                query_params,
                count=len(events),
                transport=self.STREAM_SSE_TRANSPORT,
                message=self.STREAM_SSE_MESSAGE,
            )
            return HandlerResult(
                status_code=200,
                content_type="text/event-stream",
                body=self._build_sse_snapshot_body(events, metadata),
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Vary": "Accept",
                    "X-Accel-Buffering": "no",
                    "X-Aragora-Endpoint-State": self.STREAM_READINESS,
                    "X-Aragora-Stream-Mode": self.STREAM_MODE,
                    "X-Aragora-Stream-Transport": self.STREAM_SSE_TRANSPORT,
                },
            )

        payload = self._recent_payload(events)
        payload.update(
            self._stream_metadata(
                query_params,
                count=len(events),
                transport=self.STREAM_JSON_TRANSPORT,
                message=self.STREAM_JSON_MESSAGE,
            )
        )
        return json_response(
            payload,
            headers={
                "Vary": "Accept",
                "X-Aragora-Endpoint-State": self.STREAM_READINESS,
                "X-Aragora-Stream-Mode": self.STREAM_MODE,
                "X-Aragora-Stream-Transport": self.STREAM_JSON_TRANSPORT,
            },
        )

    def _get_recent_events(self, query_params: dict[str, Any]) -> list[Any]:
        """Return filtered recent spectate events from the bridge buffer."""
        try:
            from aragora.spectate.ws_bridge import get_spectate_bridge

            bridge = get_spectate_bridge()

            count_str = query_params.get("count", "50") if query_params else "50"
            try:
                count = min(int(count_str), 500)
            except (ValueError, TypeError):
                count = 50

            events = bridge.get_recent_events(count)

            debate_id = query_params.get("debate_id") if query_params else None
            pipeline_id = query_params.get("pipeline_id") if query_params else None

            if debate_id:
                events = [e for e in events if e.debate_id == debate_id]
            if pipeline_id:
                events = [e for e in events if e.pipeline_id == pipeline_id]

            return events
        except ImportError:
            return []

    def _recent_payload(self, events: list[Any]) -> dict[str, Any]:
        """Build the recent-events payload shared by snapshot endpoints."""
        return {
            "events": [e.to_dict() for e in events],
            "count": len(events),
        }

    def _stream_metadata(
        self,
        query_params: dict[str, Any],
        *,
        count: int,
        transport: str,
        message: str,
    ) -> dict[str, Any]:
        """Build stream metadata shared across JSON and SSE snapshot responses."""
        metadata: dict[str, Any] = {
            "mode": self.STREAM_MODE,
            "transport": transport,
            "readiness": self.STREAM_READINESS,
            "streaming_ready": False,
            "message": message,
            "count": count,
        }
        if query_params.get("debate_id"):
            metadata["debate_id"] = query_params["debate_id"]
        if query_params.get("pipeline_id"):
            metadata["pipeline_id"] = query_params["pipeline_id"]
        return metadata

    def _build_sse_snapshot_body(self, events: list[Any], metadata: dict[str, Any]) -> bytes:
        """Serialize buffered spectate events into a finite SSE snapshot body."""
        frames = [_sse_frame("connected", metadata)]
        for event in events:
            event_type = getattr(event, "event_type", None) or "event"
            frames.append(_sse_frame(event_type, event.to_dict()))
        frames.append(_sse_frame("snapshot_complete", metadata))
        return "".join(frames).encode("utf-8")

    def _wants_sse(self, query_params: dict[str, Any], handler: Any) -> bool:
        """Return True when the caller requested an SSE response."""
        if (query_params.get("format") or "").lower() == "sse":
            return True
        headers = getattr(handler, "headers", {}) or {}
        accept = headers.get("Accept") or headers.get("accept") or ""
        return "text/event-stream" in accept

    def _handle_status(self, handler: Any) -> HandlerResult:
        """GET /api/v1/spectate/status -- bridge status."""
        try:
            from aragora.spectate.ws_bridge import get_spectate_bridge

            bridge = get_spectate_bridge()
            summary = _summarize_bridge_activity(
                bridge.get_recent_events(_STATUS_ACTIVITY_SCAN_LIMIT),
                bridge_running=bridge.running,
            )
            user = self.get_current_user(handler)
            permissions = set(getattr(user, "permissions", []) or []) if user is not None else set()
            roles = set(getattr(user, "roles", []) or []) if user is not None else set()
            role = getattr(user, "role", None) if user is not None else None
            can_view_live_debates = user is not None and (
                "debates:read" in permissions
                or "admin" in permissions
                or "admin" in roles
                or role == "admin"
            )
            if not can_view_live_debates:
                summary = _redact_live_debate_details(summary)
            return json_response(
                {
                    "active": bridge.running,
                    "subscribers": bridge.subscriber_count,
                    "buffer_size": bridge.buffer_size,
                    **summary,
                }
            )
        except ImportError:
            return json_response(
                {
                    "active": False,
                    "subscribers": 0,
                    "buffer_size": 0,
                    "bridge_state": "inactive",
                    "last_event_at": None,
                    "activity_age_seconds": None,
                    "recent_activity_window_seconds": _RECENT_ACTIVITY_WINDOW_SECONDS,
                    "recent_event_count": 0,
                    "live_debate_count": 0,
                    "live_debate_ids": [],
                    "live_debates": [],
                    "unattributed_recent_event_count": 0,
                }
            )
