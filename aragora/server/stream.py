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
    """Types of events emitted during debates and nomic loop execution."""
    # Debate events
    DEBATE_START = "debate_start"
    ROUND_START = "round_start"
    AGENT_MESSAGE = "agent_message"
    CRITIQUE = "critique"
    VOTE = "vote"
    CONSENSUS = "consensus"
    DEBATE_END = "debate_end"

    # Nomic loop events
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_RETRY = "task_retry"
    VERIFICATION_START = "verification_start"
    VERIFICATION_RESULT = "verification_result"
    COMMIT = "commit"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    ERROR = "error"
    LOG_MESSAGE = "log_message"

    # Multi-loop management events
    LOOP_REGISTER = "loop_register"      # New loop instance started
    LOOP_UNREGISTER = "loop_unregister"  # Loop instance ended
    LOOP_LIST = "loop_list"              # List of active loops (sent on connect)


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


@dataclass
class LoopInstance:
    """Represents an active nomic loop instance."""
    loop_id: str
    name: str
    started_at: float
    cycle: int = 0
    phase: str = "starting"
    path: str = ""


class DebateStreamServer:
    """
    WebSocket server broadcasting debate events to connected clients.

    Supports multiple concurrent nomic loop instances with view switching.

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
        # Multi-loop tracking
        self.active_loops: dict[str, LoopInstance] = {}  # loop_id -> LoopInstance

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

    def register_loop(self, loop_id: str, name: str, path: str = "") -> None:
        """Register a new nomic loop instance."""
        instance = LoopInstance(
            loop_id=loop_id,
            name=name,
            started_at=time.time(),
            path=path,
        )
        self.active_loops[loop_id] = instance
        # Emit registration event
        self._emitter.emit(StreamEvent(
            type=StreamEventType.LOOP_REGISTER,
            data={
                "loop_id": loop_id,
                "name": name,
                "started_at": instance.started_at,
                "path": path,
                "active_loops": len(self.active_loops),
            },
        ))

    def unregister_loop(self, loop_id: str) -> None:
        """Unregister a nomic loop instance."""
        if loop_id in self.active_loops:
            del self.active_loops[loop_id]
            # Emit unregistration event
            self._emitter.emit(StreamEvent(
                type=StreamEventType.LOOP_UNREGISTER,
                data={
                    "loop_id": loop_id,
                    "active_loops": len(self.active_loops),
                },
            ))

    def update_loop_state(self, loop_id: str, cycle: int = None, phase: str = None) -> None:
        """Update the state of an active loop instance."""
        if loop_id in self.active_loops:
            if cycle is not None:
                self.active_loops[loop_id].cycle = cycle
            if phase is not None:
                self.active_loops[loop_id].phase = phase

    def get_loop_list(self) -> list[dict]:
        """Get list of active loops for client sync."""
        return [
            {
                "loop_id": loop.loop_id,
                "name": loop.name,
                "started_at": loop.started_at,
                "cycle": loop.cycle,
                "phase": loop.phase,
                "path": loop.path,
            }
            for loop in self.active_loops.values()
        ]

    async def handler(self, websocket) -> None:
        """Handle a WebSocket connection."""
        self.clients.add(websocket)
        try:
            # Send list of active loops
            await websocket.send(json.dumps({
                "type": "loop_list",
                "data": {
                    "loops": self.get_loop_list(),
                    "count": len(self.active_loops),
                }
            }))

            # Send current debate state if one is in progress
            if self.current_debate:
                await websocket.send(json.dumps({
                    "type": "sync",
                    "data": self.current_debate
                }))

            # Keep connection alive, handle incoming messages if needed
            async for message in websocket:
                # Handle client requests (e.g., switch active loop view)
                try:
                    data = json.loads(message)
                    if data.get("type") == "get_loops":
                        await websocket.send(json.dumps({
                            "type": "loop_list",
                            "data": {
                                "loops": self.get_loop_list(),
                                "count": len(self.active_loops),
                            }
                        }))
                except json.JSONDecodeError:
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
        agent: str, target: str, issues: list[str], severity: float, round_num: int,
        full_content: str = None
    ) -> None:
        emitter.emit(StreamEvent(
            type=StreamEventType.CRITIQUE,
            data={
                "target": target,
                "issues": issues,  # Full issue list
                "severity": severity,
                "content": full_content or "\n".join(f"• {issue}" for issue in issues),
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
                "answer": answer,  # Full answer - no truncation
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
