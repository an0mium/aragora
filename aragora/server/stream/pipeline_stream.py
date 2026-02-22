"""
WebSocket stream handler for idea-to-execution pipeline events.

Provides real-time streaming of pipeline stage progress, goal extraction,
workflow generation, and completion/failure events to connected clients.

Follows the AutonomousStreamEmitter pattern from autonomous_stream.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from aragora.events.types import StreamEvent, StreamEventType

logger = logging.getLogger(__name__)


@dataclass
class PipelineStreamClient:
    """A connected WebSocket client for pipeline events."""

    ws: web.WebSocketResponse
    client_id: str
    pipeline_id: str
    connected_at: float = field(default_factory=time.time)
    subscriptions: set[str] = field(default_factory=set)


class PipelineStreamEmitter:
    """Emitter for idea-to-execution pipeline events.

    Broadcasts events to connected WebSocket clients, filtered by
    pipeline_id so each client only receives events for the pipeline
    they are observing.
    """

    def __init__(self) -> None:
        self._clients: dict[str, PipelineStreamClient] = {}
        self._event_history: dict[str, list[dict[str, Any]]] = {}
        self._max_history_per_pipeline = 500
        self._client_counter = 0

    def add_client(
        self,
        ws: web.WebSocketResponse,
        pipeline_id: str,
        subscriptions: set[str] | None = None,
    ) -> str:
        """Add a new WebSocket client watching a specific pipeline."""
        self._client_counter += 1
        client_id = f"pipe_{self._client_counter}_{int(time.time())}"
        self._clients[client_id] = PipelineStreamClient(
            ws=ws,
            client_id=client_id,
            pipeline_id=pipeline_id,
            subscriptions=subscriptions or set(),
        )
        logger.info("Pipeline stream client connected: %s for pipeline %s", client_id, pipeline_id)
        return client_id

    def remove_client(self, client_id: str) -> None:
        """Remove a WebSocket client."""
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info("Pipeline stream client disconnected: %s", client_id)

    async def emit(
        self,
        pipeline_id: str,
        event_type: StreamEventType,
        data: dict[str, Any],
    ) -> None:
        """Emit an event to all clients watching the given pipeline."""
        event = StreamEvent(type=event_type, data={"pipeline_id": pipeline_id, **data})
        event_dict = event.to_dict()

        # Store in per-pipeline history
        history = self._event_history.setdefault(pipeline_id, [])
        history.append(event_dict)
        if len(history) > self._max_history_per_pipeline:
            self._event_history[pipeline_id] = history[-self._max_history_per_pipeline:]

        event_json = json.dumps(event_dict)
        disconnected: list[str] = []

        for client_id, client in self._clients.items():
            # Only send to clients watching this pipeline
            if client.pipeline_id != pipeline_id:
                continue
            # Check subscription filter
            if client.subscriptions and event_type.value not in client.subscriptions:
                continue
            try:
                await client.ws.send_str(event_json)
            except (ConnectionError, OSError, RuntimeError):
                disconnected.append(client_id)

        for client_id in disconnected:
            self.remove_client(client_id)

    # Convenience methods for common events

    async def emit_started(
        self, pipeline_id: str, config: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_STARTED, {
            "config": config or {},
        })

    async def emit_stage_started(
        self, pipeline_id: str, stage_name: str, config: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_STAGE_STARTED, {
            "stage": stage_name, "config": config or {},
        })

    async def emit_stage_completed(
        self, pipeline_id: str, stage_name: str, summary: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_STAGE_COMPLETED, {
            "stage": stage_name, "summary": summary or {},
        })

    async def emit_graph_updated(
        self, pipeline_id: str, react_flow_data: dict[str, Any],
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_GRAPH_UPDATED, {
            "graph": react_flow_data,
        })

    async def emit_goal_extracted(
        self, pipeline_id: str, goal_dict: dict[str, Any],
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_GOAL_EXTRACTED, {
            "goal": goal_dict,
        })

    async def emit_workflow_generated(
        self, pipeline_id: str, workflow_dict: dict[str, Any],
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_WORKFLOW_GENERATED, {
            "workflow": workflow_dict,
        })

    async def emit_step_progress(
        self, pipeline_id: str, step_name: str, progress: float,
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_STEP_PROGRESS, {
            "step": step_name, "progress": progress,
        })

    async def emit_node_added(
        self,
        pipeline_id: str,
        stage: str,
        node_id: str,
        node_type: str,
        label: str,
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_NODE_ADDED, {
            "stage": stage,
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "added_at": time.time(),
        })

    async def emit_transition_pending(
        self,
        pipeline_id: str,
        from_stage: str,
        to_stage: str,
        confidence: float,
        ai_rationale: str,
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_TRANSITION_PENDING, {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "confidence": confidence,
            "ai_rationale": ai_rationale,
            "pending_at": time.time(),
        })

    async def emit_completed(
        self, pipeline_id: str, receipt_dict: dict[str, Any] | None = None,
    ) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_COMPLETED, {
            "receipt": receipt_dict,
        })

    async def emit_failed(self, pipeline_id: str, error: str) -> None:
        await self.emit(pipeline_id, StreamEventType.PIPELINE_FAILED, {
            "error": error,
        })

    def get_history(self, pipeline_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent event history for a pipeline."""
        history = self._event_history.get(pipeline_id, [])
        return history[-limit:]

    def as_event_callback(self, pipeline_id: str) -> Any:
        """Return a sync callback suitable for PipelineConfig.event_callback.

        Creates a closure that schedules async emit calls on the running loop.
        """
        def callback(event_type: str, data: dict[str, Any]) -> None:
            type_map = {
                "started": StreamEventType.PIPELINE_STARTED,
                "stage_started": StreamEventType.PIPELINE_STAGE_STARTED,
                "stage_completed": StreamEventType.PIPELINE_STAGE_COMPLETED,
                "graph_updated": StreamEventType.PIPELINE_GRAPH_UPDATED,
                "goal_extracted": StreamEventType.PIPELINE_GOAL_EXTRACTED,
                "workflow_generated": StreamEventType.PIPELINE_WORKFLOW_GENERATED,
                "step_progress": StreamEventType.PIPELINE_STEP_PROGRESS,
                "node_added": StreamEventType.PIPELINE_NODE_ADDED,
                "pipeline_node_added": StreamEventType.PIPELINE_NODE_ADDED,
                "transition_pending": StreamEventType.PIPELINE_TRANSITION_PENDING,
                "completed": StreamEventType.PIPELINE_COMPLETED,
                "failed": StreamEventType.PIPELINE_FAILED,
            }
            st = type_map.get(event_type, StreamEventType.PIPELINE_STEP_PROGRESS)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.emit(pipeline_id, st, data))
            except RuntimeError:
                pass

        return callback

    @property
    def client_count(self) -> int:
        return len(self._clients)


# Global emitter instance
_pipeline_emitter: PipelineStreamEmitter | None = None


def get_pipeline_emitter() -> PipelineStreamEmitter:
    """Get or create the global pipeline stream emitter."""
    global _pipeline_emitter
    if _pipeline_emitter is None:
        _pipeline_emitter = PipelineStreamEmitter()
    return _pipeline_emitter


def set_pipeline_emitter(emitter: PipelineStreamEmitter | None) -> None:
    """Set the global pipeline stream emitter (for testing)."""
    global _pipeline_emitter
    _pipeline_emitter = emitter


async def pipeline_websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler for pipeline events.

    Endpoint: /ws/pipeline

    Query params:
        pipeline_id: Required — the pipeline to observe
        subscribe: Comma-separated event types to filter (optional)

    Incoming messages:
        {"type": "ping"}
        {"type": "subscribe", "events": ["pipeline_completed"]}
        {"type": "unsubscribe", "events": ["pipeline_step_progress"]}
        {"type": "get_history", "limit": 50}
    """
    from aiohttp import WSMsgType

    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    pipeline_id = request.query.get("pipeline_id", "")
    if not pipeline_id:
        await ws.send_json({"type": "error", "message": "Missing pipeline_id query param"})
        await ws.close()
        return ws

    subscribe_param = request.query.get("subscribe", "")
    subscriptions = set(subscribe_param.split(",")) if subscribe_param else set()

    emitter = get_pipeline_emitter()
    client_id = emitter.add_client(ws, pipeline_id, subscriptions)

    await ws.send_json({
        "type": "connected",
        "client_id": client_id,
        "pipeline_id": pipeline_id,
        "subscriptions": list(subscriptions),
        "timestamp": time.time(),
    })

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await ws.send_json({"type": "pong", "timestamp": time.time()})

                    elif msg_type == "subscribe":
                        events = data.get("events", [])
                        if client_id in emitter._clients:
                            emitter._clients[client_id].subscriptions.update(events)
                            await ws.send_json({"type": "subscribed", "events": events})

                    elif msg_type == "unsubscribe":
                        events = data.get("events", [])
                        if client_id in emitter._clients:
                            emitter._clients[client_id].subscriptions.difference_update(events)
                            await ws.send_json({"type": "unsubscribed", "events": events})

                    elif msg_type == "get_history":
                        limit = data.get("limit", 100)
                        history = emitter.get_history(pipeline_id, limit)
                        await ws.send_json({
                            "type": "history",
                            "events": history,
                            "count": len(history),
                        })

                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})

            elif msg.type == WSMsgType.ERROR:
                logger.error("Pipeline WS error: %s", ws.exception())
                break
    finally:
        emitter.remove_client(client_id)

    return ws


def register_pipeline_stream_routes(app: web.Application) -> None:
    """Register the pipeline stream WebSocket route."""
    app.router.add_get("/ws/pipeline", pipeline_websocket_handler)


__all__ = [
    "PipelineStreamClient",
    "PipelineStreamEmitter",
    "get_pipeline_emitter",
    "set_pipeline_emitter",
    "pipeline_websocket_handler",
    "register_pipeline_stream_routes",
]
