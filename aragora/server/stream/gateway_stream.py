"""
Gateway WebSocket Stream Server.

Provides real-time streaming of gateway events including:
- External agent response chunks
- Capability usage notifications
- Error events
- Completion events
- Debate progress updates
- Verification agent status updates

Usage:
    from aragora.server.stream.gateway_stream import GatewayStreamServer

    server = GatewayStreamServer()
    await server.broadcast_agent_chunk(agent_name, chunk)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class GatewayEventType(Enum):
    """Types of gateway stream events."""

    AGENT_RESPONSE_CHUNK = "agent_response_chunk"
    AGENT_CAPABILITY_USED = "agent_capability_used"
    AGENT_ERROR = "agent_error"
    AGENT_COMPLETE = "agent_complete"
    DEBATE_PROGRESS = "debate_progress"
    VERIFICATION_UPDATE = "verification_update"


@dataclass
class GatewayEvent:
    """A gateway stream event."""

    event_type: GatewayEventType
    agent_name: str
    data: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    debate_id: str | None = None
    sequence: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.event_type.value,
            "agent": self.agent_name,
            "data": self.data,
            "timestamp": self.timestamp,
            "debate_id": self.debate_id,
            "sequence": self.sequence,
        }


class GatewayStreamServer:
    """WebSocket server for gateway event streaming."""

    def __init__(self):
        """Initialize the gateway stream server."""
        self._subscribers: dict[str, set[Callable]] = {}  # debate_id -> callbacks
        self._global_subscribers: set[Callable] = set()
        self._sequence: int = 0
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        callback: Callable[[GatewayEvent], Any],
        debate_id: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to gateway events.

        Args:
            callback: Function to call with each event. Can be sync or async.
            debate_id: Optional debate ID to subscribe to specific debate events.
                       If None, subscribes to all events.

        Returns:
            An unsubscribe function that removes the subscription.
        """
        async with self._lock:
            if debate_id is None:
                self._global_subscribers.add(callback)
            else:
                if debate_id not in self._subscribers:
                    self._subscribers[debate_id] = set()
                self._subscribers[debate_id].add(callback)

        def unsubscribe() -> None:
            """Remove this subscription."""
            if debate_id is None:
                self._global_subscribers.discard(callback)
            else:
                if debate_id in self._subscribers:
                    self._subscribers[debate_id].discard(callback)
                    # Clean up empty sets
                    if not self._subscribers[debate_id]:
                        del self._subscribers[debate_id]

        return unsubscribe

    async def broadcast(self, event: GatewayEvent) -> int:
        """Broadcast event to all relevant subscribers.

        Args:
            event: The event to broadcast.

        Returns:
            Count of subscribers that received the event.
        """
        async with self._lock:
            self._sequence += 1
            event.sequence = self._sequence

            # Collect all relevant callbacks
            callbacks: list[Callable] = list(self._global_subscribers)

            # Add debate-specific subscribers if applicable
            if event.debate_id and event.debate_id in self._subscribers:
                callbacks.extend(self._subscribers[event.debate_id])

        sent_count = 0
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Error in gateway stream callback: {e}")
                # Continue to other subscribers even if one fails

        return sent_count

    async def broadcast_agent_chunk(
        self,
        agent_name: str,
        chunk: str,
        debate_id: str | None = None,
    ) -> int:
        """Broadcast a response chunk from an agent.

        Args:
            agent_name: Name of the agent sending the chunk.
            chunk: The text chunk content.
            debate_id: Optional debate ID.

        Returns:
            Count of subscribers that received the event.
        """
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_RESPONSE_CHUNK,
            agent_name=agent_name,
            data={"chunk": chunk},
            debate_id=debate_id,
        )
        return await self.broadcast(event)

    async def broadcast_capability_used(
        self,
        agent_name: str,
        capability: str,
        details: dict | None = None,
        debate_id: str | None = None,
    ) -> int:
        """Broadcast that an agent used a capability.

        Args:
            agent_name: Name of the agent.
            capability: Name of the capability/tool used.
            details: Optional additional details about the capability usage.
            debate_id: Optional debate ID.

        Returns:
            Count of subscribers that received the event.
        """
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_CAPABILITY_USED,
            agent_name=agent_name,
            data={"capability": capability, "details": details or {}},
            debate_id=debate_id,
        )
        return await self.broadcast(event)

    async def broadcast_agent_error(
        self,
        agent_name: str,
        error: str,
        error_code: str | None = None,
        debate_id: str | None = None,
    ) -> int:
        """Broadcast an agent error.

        Args:
            agent_name: Name of the agent that encountered the error.
            error: Error message.
            error_code: Optional error code.
            debate_id: Optional debate ID.

        Returns:
            Count of subscribers that received the event.
        """
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_ERROR,
            agent_name=agent_name,
            data={"error": error, "error_code": error_code},
            debate_id=debate_id,
        )
        return await self.broadcast(event)

    async def broadcast_agent_complete(
        self,
        agent_name: str,
        result: dict,
        debate_id: str | None = None,
    ) -> int:
        """Broadcast agent completion.

        Args:
            agent_name: Name of the agent that completed.
            result: Result data from the agent.
            debate_id: Optional debate ID.

        Returns:
            Count of subscribers that received the event.
        """
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_COMPLETE,
            agent_name=agent_name,
            data={"result": result},
            debate_id=debate_id,
        )
        return await self.broadcast(event)

    async def broadcast_debate_progress(
        self,
        debate_id: str,
        phase: str,
        progress: float,
        message: str = "",
    ) -> int:
        """Broadcast debate progress update.

        Args:
            debate_id: ID of the debate.
            phase: Current phase name.
            progress: Progress as a float between 0.0 and 1.0.
            message: Optional progress message.

        Returns:
            Count of subscribers that received the event.
        """
        event = GatewayEvent(
            event_type=GatewayEventType.DEBATE_PROGRESS,
            agent_name="system",
            data={"phase": phase, "progress": progress, "message": message},
            debate_id=debate_id,
        )
        return await self.broadcast(event)

    async def broadcast_verification_update(
        self,
        debate_id: str,
        verifier_name: str,
        status: str,
        critique: str | None = None,
    ) -> int:
        """Broadcast verification agent update.

        Args:
            debate_id: ID of the debate.
            verifier_name: Name of the verification agent.
            status: Status of the verification (e.g., "started", "completed", "failed").
            critique: Optional critique text from the verifier.

        Returns:
            Count of subscribers that received the event.
        """
        event = GatewayEvent(
            event_type=GatewayEventType.VERIFICATION_UPDATE,
            agent_name=verifier_name,
            data={"status": status, "critique": critique},
            debate_id=debate_id,
        )
        return await self.broadcast(event)

    def get_subscriber_count(self, debate_id: str | None = None) -> int:
        """Get count of subscribers.

        Args:
            debate_id: If provided, count only subscribers for this debate.
                       If None, count all subscribers (global + all debate-specific).

        Returns:
            Number of subscribers.
        """
        if debate_id is None:
            # Count all subscribers
            total = len(self._global_subscribers)
            for subs in self._subscribers.values():
                total += len(subs)
            return total
        else:
            # Count global + debate-specific
            count = len(self._global_subscribers)
            if debate_id in self._subscribers:
                count += len(self._subscribers[debate_id])
            return count


# Global instance
_gateway_stream_server: GatewayStreamServer | None = None


def get_gateway_stream_server() -> GatewayStreamServer:
    """Get global gateway stream server instance."""
    global _gateway_stream_server
    if _gateway_stream_server is None:
        _gateway_stream_server = GatewayStreamServer()
        _maybe_register_tool_capture(_gateway_stream_server)
    return _gateway_stream_server


def set_gateway_stream_server(server: GatewayStreamServer) -> None:
    """Set the global gateway stream server instance.

    Args:
        server: The server instance to set as global.
    """
    global _gateway_stream_server
    _gateway_stream_server = server


_tool_capture_registered = False


def _maybe_register_tool_capture(server: GatewayStreamServer) -> None:
    """Register tool usage capture callback if enabled."""
    global _tool_capture_registered
    if _tool_capture_registered:
        return
    if os.environ.get("ARAGORA_MEMORY_CAPTURE_ENABLED", "false").lower() != "true":
        return

    try:
        from aragora.memory.capture import ToolMemoryCapture

        capture = ToolMemoryCapture()
    except Exception:
        logger.debug("ToolMemoryCapture unavailable, tool usage capture disabled", exc_info=True)
        return

    async def _capture_tool_event(event: GatewayEvent) -> None:
        if event.event_type != GatewayEventType.AGENT_CAPABILITY_USED:
            return

        try:
            from aragora.memory.continuum import MemoryTier, get_continuum_memory
        except Exception:
            logger.debug("Continuum memory unavailable for tool event capture", exc_info=True)
            return

        tool_name = event.data.get("capability")
        if not capture.should_capture(tool_name):
            return

        def _store() -> None:
            memory = get_continuum_memory()
            try:
                tier_value = capture.config.tier
                tier = MemoryTier(tier_value) if tier_value else MemoryTier.FAST
            except Exception:
                logger.debug("Invalid memory tier in capture config, defaulting to FAST", exc_info=True)
                tier = MemoryTier.FAST

            content = capture.format_content(
                tool_name=str(tool_name),
                agent_name=event.agent_name,
                debate_id=event.debate_id,
                details=event.data.get("details"),
            )
            metadata = {
                "type": "tool_usage",
                "tool": tool_name,
                "agent": event.agent_name,
                "debate_id": event.debate_id,
                "timestamp": event.timestamp,
                "source": "gateway_stream",
            }
            memory.add(
                id=f"tool:{uuid.uuid4().hex[:12]}",
                content=content,
                tier=tier,
                importance=capture.config.importance,
                metadata=metadata,
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _store)

    async def _register() -> None:
        await server.subscribe(_capture_tool_event)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_register())
    except RuntimeError:
        asyncio.run(_register())

    _tool_capture_registered = True


__all__ = [
    "GatewayStreamServer",
    "GatewayEvent",
    "GatewayEventType",
    "get_gateway_stream_server",
    "set_gateway_stream_server",
]
