"""
Real-time debate streaming via WebSocket.

The SyncEventEmitter bridges synchronous Arena code with async WebSocket broadcasts.
Events are queued synchronously and consumed by an async drain loop.
"""

import asyncio
import json
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any


class StreamEventType(Enum):
    """Types of events emitted during a debate."""
    DEBATE_START = "debate_start"
    ROUND_START = "round_start"
    AGENT_MESSAGE = "agent_message"
    CRITIQUE = "critique"
    VOTE = "vote"
    CONSENSUS = "consensus"
    DEBATE_END = "debate_end"


@dataclass
class StreamEvent:
    """A single event in the debate stream."""
    type: StreamEventType
    data: dict
    timestamp: float = field(default_factory=time.time)
    round: int = 0
    agent: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "round": self.round,
            "agent": self.agent,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class SyncEventEmitter:
    """
    Thread-safe event emitter bridging sync Arena code with async WebSocket.

    Events are queued synchronously via emit() and consumed by async drain().
    This pattern avoids needing to rewrite Arena to be fully async.
    """

    def __init__(self):
        self._queue: queue.Queue[StreamEvent] = queue.Queue()
        self._subscribers: list[Callable[[StreamEvent], None]] = []

    def emit(self, event: StreamEvent) -> None:
        """Emit event (safe to call from sync code)."""
        self._queue.put(event)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[StreamEvent], None]) -> None:
        """Add synchronous subscriber for immediate event handling."""
        self._subscribers.append(callback)

    def drain(self) -> list[StreamEvent]:
        """Get all queued events (non-blocking)."""
        events = []
        try:
            while True:
                events.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return events


class DebateStreamServer:
    """
    WebSocket server broadcasting debate events to connected clients.

    Usage:
        server = DebateStreamServer(port=8765)
        hooks = create_arena_hooks(server.emitter)
        arena = Arena(env, agents, event_hooks=hooks)

        # In async context:
        asyncio.create_task(server.start())
        await arena.run()
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: set = set()
        self.current_debate: Optional[dict] = None
        self._emitter = SyncEventEmitter()
        self._running = False

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter for Arena hooks."""
        return self._emitter

    async def broadcast(self, event: StreamEvent) -> None:
        """Send event to all connected clients."""
        if not self.clients:
            return

        message = event.to_json()
        disconnected = set()

        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                disconnected.add(client)

        self.clients -= disconnected

    async def _drain_loop(self) -> None:
        """Background task that drains the emitter queue and broadcasts."""
        while self._running:
            for event in self._emitter.drain():
                await self.broadcast(event)
            await asyncio.sleep(0.05)

    async def handler(self, websocket) -> None:
        """Handle a WebSocket connection."""
        self.clients.add(websocket)
        try:
            # Send current debate state if one is in progress
            if self.current_debate:
                await websocket.send(json.dumps({
                    "type": "sync",
                    "data": self.current_debate
                }))

            # Keep connection alive, handle incoming messages if needed
            async for message in websocket:
                # Currently we don't expect client messages, but could add commands
                pass
        finally:
            self.clients.discard(websocket)

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        self._running = True
        asyncio.create_task(self._drain_loop())

        async with websockets.serve(self.handler, self.host, self.port):
            print(f"WebSocket server: ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    def stop(self) -> None:
        """Stop the server."""
        self._running = False


def create_arena_hooks(emitter: SyncEventEmitter) -> dict[str, Callable]:
    """
    Create hook functions for Arena event emission.

    These hooks are called synchronously by Arena at key points during debate.
    They emit events to the emitter queue for async WebSocket broadcast.

    Returns:
        dict of hook name -> callback function
    """

    def on_debate_start(task: str, agents: list[str]) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": task, "agents": agents},
        ))

    def on_round_start(round_num: int) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.ROUND_START,
            data={"round": round_num},
            round=round_num,
        ))

    def on_message(agent: str, content: str, role: str, round_num: int) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"content": content, "role": role},
            round=round_num,
            agent=agent,
        ))

    def on_critique(
        agent: str, target: str, issues: list[str], severity: float, round_num: int
    ) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.CRITIQUE,
            data={
                "target": target,
                "issues": issues[:3],  # Limit for brevity
                "severity": severity,
            },
            round=round_num,
            agent=agent,
        ))

    def on_vote(agent: str, vote: str, confidence: float) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.VOTE,
            data={"vote": vote, "confidence": confidence},
            agent=agent,
        ))

    def on_consensus(reached: bool, confidence: float, answer: str) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "reached": reached,
                "confidence": confidence,
                "answer": answer[:1000],  # Truncate for streaming
            },
        ))

    def on_debate_end(duration: float, rounds: int) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.DEBATE_END,
            data={"duration": duration, "rounds": rounds},
        ))

    return {
        "on_debate_start": on_debate_start,
        "on_round_start": on_round_start,
        "on_message": on_message,
        "on_critique": on_critique,
        "on_vote": on_vote,
        "on_consensus": on_consensus,
        "on_debate_end": on_debate_end,
    }
