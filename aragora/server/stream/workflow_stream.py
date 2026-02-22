"""
WebSocket stream handler for WorkflowEngine execution events.

Maps WORKFLOW_STEP_START/COMPLETE/FAILED events to canvas node state
updates, enabling real-time visualization of workflow DAG execution.

Follows the PipelineStreamEmitter pattern from pipeline_stream.py.
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
class WorkflowStreamClient:
    """A connected WebSocket client for workflow events."""

    ws: web.WebSocketResponse
    client_id: str
    workflow_id: str
    connected_at: float = field(default_factory=time.time)


class WorkflowStreamEmitter:
    """Emitter for WorkflowEngine execution events.

    Broadcasts step-level events to connected WebSocket clients,
    filtered by workflow_id.
    """

    def __init__(self) -> None:
        self._clients: dict[str, WorkflowStreamClient] = {}
        self._event_history: dict[str, list[dict[str, Any]]] = {}
        self._max_history = 500
        self._counter = 0

    def add_client(
        self, ws: web.WebSocketResponse, workflow_id: str,
    ) -> str:
        """Add a WebSocket client watching a workflow."""
        self._counter += 1
        client_id = f"wf_{self._counter}_{int(time.time())}"
        self._clients[client_id] = WorkflowStreamClient(
            ws=ws, client_id=client_id, workflow_id=workflow_id,
        )
        logger.info("Workflow stream client connected: %s for %s", client_id, workflow_id)
        return client_id

    def remove_client(self, client_id: str) -> None:
        """Remove a WebSocket client."""
        self._clients.pop(client_id, None)

    async def emit(
        self,
        workflow_id: str,
        event_type: StreamEventType,
        data: dict[str, Any],
    ) -> None:
        """Emit an event to all clients watching the given workflow."""
        event = StreamEvent(type=event_type, data={"workflow_id": workflow_id, **data})
        event_dict = event.to_dict()

        history = self._event_history.setdefault(workflow_id, [])
        history.append(event_dict)
        if len(history) > self._max_history:
            self._event_history[workflow_id] = history[-self._max_history:]

        event_json = json.dumps(event_dict)
        disconnected: list[str] = []

        for cid, client in self._clients.items():
            if client.workflow_id != workflow_id:
                continue
            try:
                await client.ws.send_str(event_json)
            except (ConnectionError, OSError, RuntimeError):
                disconnected.append(cid)

        for cid in disconnected:
            self.remove_client(cid)

    async def emit_step_started(
        self, workflow_id: str, step_id: str, step_name: str,
    ) -> None:
        """Emit WORKFLOW_STEP_START event."""
        await self.emit(workflow_id, StreamEventType.PIPELINE_STEP_PROGRESS, {
            "step_id": step_id,
            "step_name": step_name,
            "status": "started",
        })

    async def emit_step_completed(
        self, workflow_id: str, step_id: str, step_name: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        """Emit WORKFLOW_STEP_COMPLETE event."""
        await self.emit(workflow_id, StreamEventType.PIPELINE_STEP_PROGRESS, {
            "step_id": step_id,
            "step_name": step_name,
            "status": "completed",
            "output": output or {},
        })

    async def emit_step_failed(
        self, workflow_id: str, step_id: str, step_name: str, error: str = "",
    ) -> None:
        """Emit WORKFLOW_STEP_FAILED event."""
        await self.emit(workflow_id, StreamEventType.PIPELINE_STEP_PROGRESS, {
            "step_id": step_id,
            "step_name": step_name,
            "status": "failed",
            "error": error,
        })

    def get_history(self, workflow_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent event history for a workflow."""
        history = self._event_history.get(workflow_id, [])
        return history[-limit:]

    @property
    def client_count(self) -> int:
        return len(self._clients)


# Module-level singleton
_workflow_emitter: WorkflowStreamEmitter | None = None


def get_workflow_emitter() -> WorkflowStreamEmitter:
    """Get or create the global WorkflowStreamEmitter."""
    global _workflow_emitter
    if _workflow_emitter is None:
        _workflow_emitter = WorkflowStreamEmitter()
    return _workflow_emitter


def set_workflow_emitter(emitter: WorkflowStreamEmitter | None) -> None:
    """Set the global workflow stream emitter (for testing)."""
    global _workflow_emitter
    _workflow_emitter = emitter


async def workflow_websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """WebSocket handler for workflow events.

    Endpoint: /ws/workflow

    Query params:
        workflow_id: Required -- the workflow to observe

    Incoming messages:
        {"type": "ping"}
        {"type": "get_history", "limit": 50}
    """
    from aiohttp import WSMsgType

    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    workflow_id = request.query.get("workflow_id", "")
    if not workflow_id:
        await ws.send_json({"type": "error", "message": "Missing workflow_id query param"})
        await ws.close()
        return ws

    emitter = get_workflow_emitter()
    client_id = emitter.add_client(ws, workflow_id)

    await ws.send_json({
        "type": "connected",
        "client_id": client_id,
        "workflow_id": workflow_id,
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

                    elif msg_type == "get_history":
                        limit = data.get("limit", 100)
                        history = emitter.get_history(workflow_id, limit)
                        await ws.send_json({
                            "type": "history",
                            "events": history,
                            "count": len(history),
                        })

                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})

            elif msg.type == WSMsgType.ERROR:
                logger.error("Workflow WS error: %s", ws.exception())
                break
    finally:
        emitter.remove_client(client_id)

    return ws


def register_workflow_stream_routes(app: web.Application) -> None:
    """Register the workflow stream WebSocket route."""
    app.router.add_get("/ws/workflow", workflow_websocket_handler)


__all__ = [
    "WorkflowStreamClient",
    "WorkflowStreamEmitter",
    "get_workflow_emitter",
    "set_workflow_emitter",
    "workflow_websocket_handler",
    "register_workflow_stream_routes",
]
