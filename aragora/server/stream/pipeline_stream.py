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
                "stage_started": StreamEventType.PIPELINE_STAGE_STARTED,
                "stage_completed": StreamEventType.PIPELINE_STAGE_COMPLETED,
                "graph_updated": StreamEventType.PIPELINE_GRAPH_UPDATED,
                "goal_extracted": StreamEventType.PIPELINE_GOAL_EXTRACTED,
                "workflow_generated": StreamEventType.PIPELINE_WORKFLOW_GENERATED,
                "step_progress": StreamEventType.PIPELINE_STEP_PROGRESS,
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


def set_pipeline_emitter(emitter: PipelineStreamEmitter) -> None:
    """Set the global pipeline stream emitter (for testing)."""
    global _pipeline_emitter
    _pipeline_emitter = emitter


__all__ = [
    "PipelineStreamClient",
    "PipelineStreamEmitter",
    "get_pipeline_emitter",
    "set_pipeline_emitter",
]
