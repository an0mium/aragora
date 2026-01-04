"""
Real-time debate streaming via WebSocket.

The SyncEventEmitter bridges synchronous Arena code with async WebSocket broadcasts.
Events are queued synchronously and consumed by an async drain loop.
"""

import asyncio
import json
import os
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any

# Allowed origins for WebSocket connections - configure via environment variable
WS_ALLOWED_ORIGINS = os.getenv("ARAGORA_ALLOWED_ORIGINS", "").split(",")
if not WS_ALLOWED_ORIGINS or WS_ALLOWED_ORIGINS == [""]:
    # Default to common development and production origins
    WS_ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://aragora.ai",
        "https://live.aragora.ai",
        "https://www.aragora.ai",
    ]

# Maximum WebSocket message size (64KB) - prevents memory exhaustion attacks
WS_MAX_MESSAGE_SIZE = int(os.getenv("ARAGORA_WS_MAX_SIZE", 65536))


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

    # Audience participation events
    USER_VOTE = "user_vote"              # Audience member voted
    USER_SUGGESTION = "user_suggestion"  # Audience member submitted suggestion
    AUDIENCE_SUMMARY = "audience_summary"  # Clustered audience input summary
    AUDIENCE_METRICS = "audience_metrics"  # Vote counts, histograms, conviction distribution

    # Memory/learning events
    MEMORY_RECALL = "memory_recall"      # Historical context retrieved from memory
    INSIGHT_EXTRACTED = "insight_extracted"  # New insight extracted from debate

    # Ranking/leaderboard events (debate consensus feature)
    MATCH_RECORDED = "match_recorded"    # ELO match recorded, leaderboard updated
    LEADERBOARD_UPDATE = "leaderboard_update"  # Periodic leaderboard snapshot

    # Position tracking events
    FLIP_DETECTED = "flip_detected"      # Agent position reversal detected


@dataclass
class StreamEvent:
    """A single event in the debate stream."""
    type: StreamEventType
    data: dict
    timestamp: float = field(default_factory=time.time)
    round: int = 0
    agent: str = ""
    loop_id: str = ""  # For multi-loop tracking

    def to_dict(self) -> dict:
        result = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "round": self.round,
            "agent": self.agent,
        }
        if self.loop_id:
            result["loop_id"] = self.loop_id
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class AudienceMessage:
    """A message from an audience member (vote or suggestion)."""
    type: str  # "vote" or "suggestion"
    loop_id: str  # Associated nomic loop
    payload: dict  # Message content (e.g., {"choice": "option1"} for votes)
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""  # Optional user identifier


class TokenBucket:
    """
    Token bucket rate limiter for audience message throttling.

    Allows burst traffic up to burst_size, then limits to rate_per_minute.
    Thread-safe for concurrent access.
    """

    def __init__(self, rate_per_minute: float, burst_size: int):
        """
        Initialize token bucket.

        Args:
            rate_per_minute: Token refill rate (tokens per minute)
            burst_size: Maximum tokens (bucket capacity)
        """
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)  # Start full
        self.last_refill = time.monotonic()
        self._lock = __import__('threading').Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were available and consumed, False otherwise
        """
        with self._lock:
            # Refill tokens based on elapsed time
            now = time.monotonic()
            elapsed_minutes = (now - self.last_refill) / 60.0
            refill_amount = elapsed_minutes * self.rate_per_minute
            self.tokens = min(self.burst_size, self.tokens + refill_amount)
            self.last_refill = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


def normalize_intensity(value: any, default: int = 5, min_val: int = 1, max_val: int = 10) -> int:
    """
    Safely normalize vote intensity to a clamped integer.

    Args:
        value: Raw intensity value from user input (may be string, float, None, etc.)
        default: Default intensity if value is invalid
        min_val: Minimum allowed intensity
        max_val: Maximum allowed intensity

    Returns:
        Clamped integer intensity between min_val and max_val
    """
    if value is None:
        return default

    try:
        intensity = int(float(value))
    except (ValueError, TypeError):
        return default

    return max(min_val, min(max_val, intensity))


class AudienceInbox:
    """
    Thread-safe queue for audience messages.

    Collects votes and suggestions from WebSocket clients for processing
    by the debate arena.
    """

    def __init__(self):
        self._messages: list[AudienceMessage] = []
        self._lock = __import__('threading').Lock()

    def put(self, message: AudienceMessage) -> None:
        """Add a message to the inbox (thread-safe)."""
        with self._lock:
            self._messages.append(message)

    def get_all(self) -> list[AudienceMessage]:
        """
        Drain all messages from the inbox (thread-safe).

        Returns:
            List of all queued messages, emptying the inbox
        """
        with self._lock:
            messages = self._messages.copy()
            self._messages.clear()
            return messages

    def get_summary(self, loop_id: str = None) -> dict:
        """
        Get a summary of current inbox state without draining.

        Args:
            loop_id: Optional loop ID to filter messages by (multi-tenant support)

        Returns:
            Dict with vote counts, suggestions, histograms, and conviction distribution
        """
        with self._lock:
            votes = {}
            suggestions = 0
            # Per-choice intensity histograms: {choice: {intensity: count}}
            histograms = {}
            # Global conviction distribution: {intensity: count}
            conviction_distribution = {i: 0 for i in range(1, 11)}

            for msg in self._messages:
                # Filter by loop_id if provided
                if loop_id and msg.loop_id != loop_id:
                    continue

                if msg.type == "vote":
                    choice = msg.payload.get("choice", "unknown")
                    intensity = normalize_intensity(msg.payload.get("intensity"))

                    # Basic vote count
                    votes[choice] = votes.get(choice, 0) + 1

                    # Per-choice histogram
                    if choice not in histograms:
                        histograms[choice] = {i: 0 for i in range(1, 11)}
                    histograms[choice][intensity] = histograms[choice].get(intensity, 0) + 1

                    # Global conviction distribution
                    conviction_distribution[intensity] = conviction_distribution.get(intensity, 0) + 1

                elif msg.type == "suggestion":
                    suggestions += 1

            # Calculate weighted votes using intensity
            weighted_votes = {}
            for choice, histogram in histograms.items():
                weighted_sum = sum(
                    count * (0.5 + (intensity - 1) * 0.1667)  # Linear scale: 1->0.5, 10->2.0
                    for intensity, count in histogram.items()
                )
                weighted_votes[choice] = round(weighted_sum, 2)

            return {
                "votes": votes,
                "weighted_votes": weighted_votes,
                "suggestions": suggestions,
                "total": len(self._messages) if not loop_id else sum(votes.values()) + suggestions,
                "histograms": histograms,
                "conviction_distribution": conviction_distribution,
            }


class SyncEventEmitter:
    """
    Thread-safe event emitter bridging sync Arena code with async WebSocket.

    Events are queued synchronously via emit() and consumed by async drain().
    This pattern avoids needing to rewrite Arena to be fully async.
    """

    def __init__(self, loop_id: str = ""):
        self._queue: queue.Queue[StreamEvent] = queue.Queue()
        self._subscribers: list[Callable[[StreamEvent], None]] = []
        self._loop_id = loop_id  # Default loop_id for all events

    def set_loop_id(self, loop_id: str) -> None:
        """Set the loop_id to attach to all emitted events."""
        self._loop_id = loop_id

    def emit(self, event: StreamEvent) -> None:
        """Emit event (safe to call from sync code)."""
        # Add loop_id to event if not already set
        if self._loop_id and not event.loop_id:
            event.loop_id = self._loop_id
        self._queue.put(event)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception:
                pass

    def subscribe(self, callback: Callable[[StreamEvent], None]) -> None:
        """Add synchronous subscriber for immediate event handling."""
        self._subscribers.append(callback)

    def drain(self, max_batch_size: int = 100) -> list[StreamEvent]:
        """Get queued events (non-blocking) with backpressure limit."""
        events = []
        try:
            while len(events) < max_batch_size:
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
        # Debate state caching for late joiner sync
        self.debate_states: dict[str, dict] = {}  # loop_id -> debate state
        # Audience participation
        self.audience_inbox = AudienceInbox()
        self._rate_limiters: dict[str, TokenBucket] = {}  # client_id -> TokenBucket

        # Subscribe to emitter to maintain debate states
        self._emitter.subscribe(self._update_debate_state)

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter for Arena hooks."""
        return self._emitter

    def _update_debate_state(self, event: StreamEvent) -> None:
        """Update cached debate state based on emitted events."""
        loop_id = event.loop_id
        if event.type == StreamEventType.DEBATE_START:
            self.debate_states[loop_id] = {
                "id": loop_id,
                "task": event.data["task"],
                "agents": event.data["agents"],
                "messages": [],
                "consensus_reached": False,
                "consensus_confidence": 0.0,
                "consensus_answer": "",
                "started_at": event.timestamp,
                "rounds": 0,
                "ended": False,
                "duration": 0.0,
            }
        elif event.type == StreamEventType.AGENT_MESSAGE:
            if loop_id in self.debate_states:
                state = self.debate_states[loop_id]
                state["messages"].append({
                    "agent": event.agent,
                    "role": event.data["role"],
                    "round": event.round,
                    "content": event.data["content"],
                })
                # Cap at last 1000 messages to allow full debate history without truncation
                if len(state["messages"]) > 1000:
                    state["messages"] = state["messages"][-1000:]
        elif event.type == StreamEventType.CONSENSUS:
            if loop_id in self.debate_states:
                state = self.debate_states[loop_id]
                state["consensus_reached"] = event.data["reached"]
                state["consensus_confidence"] = event.data["confidence"]
                state["consensus_answer"] = event.data["answer"]
        elif event.type == StreamEventType.DEBATE_END:
            if loop_id in self.debate_states:
                state = self.debate_states[loop_id]
                state["ended"] = True
                state["duration"] = event.data["duration"]
                state["rounds"] = event.data["rounds"]
        elif event.type == StreamEventType.LOOP_UNREGISTER:
            self.debate_states.pop(loop_id, None)

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
        """Handle a WebSocket connection with origin validation."""
        # Validate origin for security
        origin = websocket.request_headers.get("Origin", "")
        if origin and origin not in WS_ALLOWED_ORIGINS:
            # Reject connection from unauthorized origin
            await websocket.close(4003, "Origin not allowed")
            return

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

            # Send sync for each active debate
            for loop_id, state in self.debate_states.items():
                await websocket.send(json.dumps({
                    "type": "sync",
                    "data": state
                }))

            # Keep connection alive, handle incoming messages if needed
            async for message in websocket:
                # Handle client requests (e.g., switch active loop view)
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "get_loops":
                        await websocket.send(json.dumps({
                            "type": "loop_list",
                            "data": {
                                "loops": self.get_loop_list(),
                                "count": len(self.active_loops),
                            }
                        }))

                    elif msg_type in ("user_vote", "user_suggestion"):
                        # Handle audience participation
                        client_id = str(id(websocket))
                        loop_id = data.get("loop_id", "")

                        # Validate loop_id exists and is active
                        if not loop_id or loop_id not in self.active_loops:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": f"Invalid or inactive loop_id: {loop_id}"}
                            }))
                            continue

                        # Get or create rate limiter for this client
                        if client_id not in self._rate_limiters:
                            self._rate_limiters[client_id] = TokenBucket(
                                rate_per_minute=10.0,  # 10 messages per minute
                                burst_size=5  # Allow burst of 5
                            )

                        # Check rate limit
                        if not self._rate_limiters[client_id].consume(1):
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Rate limited. Please wait before submitting again."}
                            }))
                            continue

                        # Create and queue the message
                        audience_msg = AudienceMessage(
                            type="vote" if msg_type == "user_vote" else "suggestion",
                            loop_id=loop_id,
                            payload=data.get("payload", {}),
                            user_id=client_id,
                        )
                        self.audience_inbox.put(audience_msg)

                        # Emit event for dashboard visibility
                        event_type = StreamEventType.USER_VOTE if msg_type == "user_vote" else StreamEventType.USER_SUGGESTION
                        self._emitter.emit(StreamEvent(
                            type=event_type,
                            data=audience_msg.payload,
                            loop_id=loop_id,
                        ))

                        # Emit updated audience metrics after each vote (with loop_id filter)
                        if msg_type == "user_vote":
                            metrics = self.audience_inbox.get_summary(loop_id=loop_id)
                            self._emitter.emit(StreamEvent(
                                type=StreamEventType.AUDIENCE_METRICS,
                                data=metrics,
                                loop_id=loop_id,
                            ))

                        # Send acknowledgment
                        await websocket.send(json.dumps({
                            "type": "ack",
                            "data": {"message": "Message received", "msg_type": msg_type}
                        }))

                except json.JSONDecodeError:
                    pass
        except Exception:
            # Silently handle connection closed errors (normal during shutdown)
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

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            max_size=WS_MAX_MESSAGE_SIZE,
            ping_interval=30,  # Send ping every 30s
            ping_timeout=10,   # Close connection if no pong within 10s
        ):
            print(f"WebSocket server: ws://{self.host}:{self.port} (max message size: {WS_MAX_MESSAGE_SIZE} bytes)")
            await asyncio.Future()  # Run forever

    def stop(self) -> None:
        """Stop the server."""
        self._running = False

    async def graceful_shutdown(self) -> None:
        """Gracefully close all client connections."""
        self._running = False
        # Close all connected clients
        if self.clients:
            close_tasks = []
            for client in list(self.clients):
                try:
                    close_tasks.append(client.close())
                except Exception:
                    pass
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            self.clients.clear()


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
