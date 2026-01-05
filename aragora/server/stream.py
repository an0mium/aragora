"""
Real-time debate streaming via WebSocket.

The SyncEventEmitter bridges synchronous Arena code with async WebSocket broadcasts.
Events are queued synchronously and consumed by an async drain loop.

This module also supports unified HTTP+WebSocket serving on a single port via aiohttp.
"""

import asyncio
import json
import logging
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Any, Dict
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor
import uuid

# Configure module logger
logger = logging.getLogger(__name__)

# Import debate components (lazy-loaded for optional functionality)
try:
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.agents.base import create_agent
    from aragora.core import Environment
    DEBATE_AVAILABLE = True
except ImportError:
    DEBATE_AVAILABLE = False
    Arena = None
    DebateProtocol = None
    create_agent = None
    Environment = None

# Valid agent types (allowlist for security)
ALLOWED_AGENT_TYPES = frozenset({
    "codex", "claude", "openai", "gemini-cli", "grok-cli", "qwen-cli", "deepseek-cli", "kilocode",
    "gemini", "ollama", "anthropic-api", "openai-api", "grok",
    "deepseek", "deepseek-r1", "llama", "mistral", "openrouter",
})

# Maximum number of agents per debate (DoS protection)
MAX_AGENTS_PER_DEBATE = 10
MAX_CONCURRENT_DEBATES = 3

# Thread-safe debate tracking
_active_debates: dict[str, dict] = {}
_active_debates_lock = threading.Lock()
_debate_executor: Optional[ThreadPoolExecutor] = None
_debate_executor_lock = threading.Lock()


def _safe_error_message(e: Exception, context: str = "") -> str:
    """Return a sanitized error message for client responses.

    Logs the full error server-side while returning a generic message to clients.
    This prevents information disclosure of internal details like file paths,
    stack traces, or sensitive configuration.
    """
    # Log full details server-side for debugging
    logger.error(f"Error in {context}: {type(e).__name__}: {e}", exc_info=True)

    # Map common exceptions to user-friendly messages
    error_type = type(e).__name__
    if error_type in ("FileNotFoundError", "OSError"):
        return "Resource not found"
    elif error_type in ("json.JSONDecodeError", "ValueError"):
        return "Invalid data format"
    elif error_type in ("PermissionError",):
        return "Access denied"
    elif error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "Operation timed out"
    else:
        return "An error occurred"


def _wrap_agent_for_streaming(agent, emitter: 'SyncEventEmitter', debate_id: str):
    """Wrap an agent to emit token streaming events.

    If the agent has a generate_stream() method, we override its generate()
    to call generate_stream() and emit TOKEN_* events.
    """
    from datetime import datetime

    # Check if agent supports streaming
    if not hasattr(agent, 'generate_stream'):
        return agent

    # Store original generate method
    original_generate = agent.generate

    async def streaming_generate(prompt: str, context=None):
        """Streaming wrapper that emits TOKEN_* events."""
        # Emit start event
        emitter.emit(StreamEvent(
            type=StreamEventType.TOKEN_START,
            data={
                "debate_id": debate_id,
                "agent": agent.name,
                "timestamp": datetime.now().isoformat(),
            },
            agent=agent.name,
        ))

        full_response = ""
        try:
            # Stream tokens from the agent
            async for token in agent.generate_stream(prompt, context):
                full_response += token
                # Emit token delta event
                emitter.emit(StreamEvent(
                    type=StreamEventType.TOKEN_DELTA,
                    data={
                        "debate_id": debate_id,
                        "agent": agent.name,
                        "token": token,
                    },
                    agent=agent.name,
                ))

            # Emit end event
            emitter.emit(StreamEvent(
                type=StreamEventType.TOKEN_END,
                data={
                    "debate_id": debate_id,
                    "agent": agent.name,
                    "full_response": full_response,
                },
                agent=agent.name,
            ))

            return full_response

        except Exception as e:
            # Emit error as end event
            emitter.emit(StreamEvent(
                type=StreamEventType.TOKEN_END,
                data={
                    "debate_id": debate_id,
                    "agent": agent.name,
                    "error": _safe_error_message(e, f"token streaming for {agent.name}"),
                    "full_response": full_response,
                },
                agent=agent.name,
            ))
            # Fall back to non-streaming
            if full_response:
                return full_response
            return await original_generate(prompt, context)

    # Replace the generate method
    agent.generate = streaming_generate
    return agent


# Centralized CORS configuration
from aragora.server.cors_config import WS_ALLOWED_ORIGINS

# Maximum WebSocket message size (64KB) - prevents memory exhaustion attacks
WS_MAX_MESSAGE_SIZE = int(os.getenv("ARAGORA_WS_MAX_SIZE", 65536))

# Trusted proxies for X-Forwarded-For header validation
# Only trust X-Forwarded-For if request comes from these IPs
TRUSTED_PROXIES = frozenset(
    os.getenv('ARAGORA_TRUSTED_PROXIES', '127.0.0.1,::1,localhost').split(',')
)


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

    # Token streaming events (for real-time response display)
    TOKEN_START = "token_start"      # Agent begins generating response
    TOKEN_DELTA = "token_delta"      # Incremental token(s) received
    TOKEN_END = "token_end"          # Agent finished generating response

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
    AUDIENCE_DRAIN = "audience_drain"    # Audience events processed by arena

    # Memory/learning events
    MEMORY_RECALL = "memory_recall"      # Historical context retrieved from memory
    INSIGHT_EXTRACTED = "insight_extracted"  # New insight extracted from debate

    # Ranking/leaderboard events (debate consensus feature)
    MATCH_RECORDED = "match_recorded"    # ELO match recorded, leaderboard updated
    LEADERBOARD_UPDATE = "leaderboard_update"  # Periodic leaderboard snapshot
    GROUNDED_VERDICT = "grounded_verdict"  # Evidence-backed verdict with citations

    # Position tracking events
    FLIP_DETECTED = "flip_detected"      # Agent position reversal detected

    # Mood/sentiment events (Real-Time Debate Drama)
    MOOD_DETECTED = "mood_detected"      # Agent emotional state analyzed
    MOOD_SHIFT = "mood_shift"            # Significant mood change detected
    DEBATE_ENERGY = "debate_energy"      # Overall debate intensity level

    # Capability probe events (Adversarial Testing)
    PROBE_START = "probe_start"          # Probe session started for agent
    PROBE_RESULT = "probe_result"        # Individual probe result
    PROBE_COMPLETE = "probe_complete"    # All probes complete, report ready

    # Deep Audit events (Intensive Multi-Round Analysis)
    AUDIT_START = "audit_start"          # Deep audit session started
    AUDIT_ROUND = "audit_round"          # Audit round completed (1-6)
    AUDIT_FINDING = "audit_finding"      # Individual finding discovered
    AUDIT_CROSS_EXAM = "audit_cross_exam"  # Cross-examination phase
    AUDIT_VERDICT = "audit_verdict"      # Final audit verdict ready


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

    # Maximum queue size to prevent memory exhaustion (DoS protection)
    MAX_QUEUE_SIZE = 10000

    def __init__(self, loop_id: str = ""):
        self._queue: queue.Queue[StreamEvent] = queue.Queue()
        self._subscribers: list[Callable[[StreamEvent], None]] = []
        self._loop_id = loop_id  # Default loop_id for all events
        self._overflow_count = 0  # Track dropped events for monitoring

    def set_loop_id(self, loop_id: str) -> None:
        """Set the loop_id to attach to all emitted events."""
        self._loop_id = loop_id

    def emit(self, event: StreamEvent) -> None:
        """Emit event (safe to call from sync code)."""
        # Add loop_id to event if not already set
        if self._loop_id and not event.loop_id:
            event.loop_id = self._loop_id

        # Enforce queue size limit to prevent memory exhaustion
        if self._queue.qsize() >= self.MAX_QUEUE_SIZE:
            # Drop oldest event to make room (backpressure)
            try:
                self._queue.get_nowait()
                self._overflow_count += 1
            except queue.Empty:
                pass

        self._queue.put(event)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception as e:
                logger.warning(f"[stream] Subscriber callback error: {e}")

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
        # Multi-loop tracking with thread safety
        self.active_loops: dict[str, LoopInstance] = {}  # loop_id -> LoopInstance
        self._active_loops_lock = threading.Lock()
        # Debate state caching for late joiner sync
        self.debate_states: dict[str, dict] = {}  # loop_id -> debate state
        self._debate_states_lock = threading.Lock()
        # Audience participation
        self.audience_inbox = AudienceInbox()
        self._rate_limiters: dict[str, TokenBucket] = {}  # client_id -> TokenBucket
        self._rate_limiter_last_access: dict[str, float] = {}  # client_id -> last access time
        self._rate_limiters_lock = threading.Lock()
        self._rate_limiter_cleanup_counter = 0  # Counter for periodic cleanup
        self._RATE_LIMITER_TTL = 3600  # 1 hour TTL for rate limiters
        self._CLEANUP_INTERVAL = 100  # Cleanup every N accesses
        # Secure client ID mapping (cryptographically random, not memory address)
        self._client_ids: Dict[int, str] = {}  # websocket id -> secure client_id

        # Subscribe to emitter to maintain debate states
        self._emitter.subscribe(self._update_debate_state)

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter for Arena hooks."""
        return self._emitter

    def _cleanup_stale_rate_limiters(self) -> None:
        """Remove rate limiters not accessed within TTL period."""
        now = time.time()
        with self._rate_limiters_lock:
            stale_keys = [
                k for k, v in self._rate_limiter_last_access.items()
                if now - v > self._RATE_LIMITER_TTL
            ]
            for k in stale_keys:
                self._rate_limiters.pop(k, None)
                self._rate_limiter_last_access.pop(k, None)
        if stale_keys:
            print(f"[stream] Cleaned up {len(stale_keys)} stale rate limiters")

    def _update_debate_state(self, event: StreamEvent) -> None:
        """Update cached debate state based on emitted events."""
        loop_id = event.loop_id
        with self._debate_states_lock:
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

        # Update loop state for cycle/phase events (outside debate_states_lock)
        if event.type == StreamEventType.CYCLE_START:
            self.update_loop_state(loop_id, cycle=event.data.get("cycle"))
        elif event.type == StreamEventType.PHASE_START:
            self.update_loop_state(loop_id, phase=event.data.get("phase"))

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
        with self._active_loops_lock:
            self.active_loops[loop_id] = instance
            loop_count = len(self.active_loops)
        # Emit registration event
        self._emitter.emit(StreamEvent(
            type=StreamEventType.LOOP_REGISTER,
            data={
                "loop_id": loop_id,
                "name": name,
                "started_at": instance.started_at,
                "path": path,
                "active_loops": loop_count,
            },
        ))

    def unregister_loop(self, loop_id: str) -> None:
        """Unregister a nomic loop instance."""
        with self._active_loops_lock:
            if loop_id in self.active_loops:
                del self.active_loops[loop_id]
                loop_count = len(self.active_loops)
            else:
                return  # Loop not found, nothing to unregister
        # Emit unregistration event
        self._emitter.emit(StreamEvent(
            type=StreamEventType.LOOP_UNREGISTER,
            data={
                "loop_id": loop_id,
                "active_loops": loop_count,
            },
        ))

    def update_loop_state(self, loop_id: str, cycle: int = None, phase: str = None) -> None:
        """Update the state of an active loop instance."""
        with self._active_loops_lock:
            if loop_id in self.active_loops:
                if cycle is not None:
                    self.active_loops[loop_id].cycle = cycle
                if phase is not None:
                    self.active_loops[loop_id].phase = phase

    def get_loop_list(self) -> list[dict]:
        """Get list of active loops for client sync."""
        with self._active_loops_lock:
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
        # Validate origin for security (handle different websockets library versions)
        try:
            # Try newer websockets API first
            if hasattr(websocket, 'request') and hasattr(websocket.request, 'headers'):
                origin = websocket.request.headers.get("Origin", "")
            elif hasattr(websocket, 'request_headers'):
                origin = websocket.request_headers.get("Origin", "")
            else:
                origin = ""
        except Exception:
            origin = ""

        if origin and origin not in WS_ALLOWED_ORIGINS:
            # Reject connection from unauthorized origin
            await websocket.close(4003, "Origin not allowed")
            return

        # Generate cryptographically secure client ID (not predictable memory address)
        ws_id = id(websocket)
        client_id = secrets.token_urlsafe(16)
        self._client_ids[ws_id] = client_id

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
                    # Validate message size before parsing (DoS protection)
                    if len(message) > WS_MAX_MESSAGE_SIZE:
                        logger.warning(f"[ws] Message too large from client: {len(message)} bytes")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "data": {"message": "Message too large"}
                        }))
                        continue

                    # Parse JSON with timeout protection (prevents CPU-bound DoS)
                    try:
                        data = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, json.loads, message),
                            timeout=5.0  # 5 second timeout for JSON parsing
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[ws] JSON parsing timed out - possible DoS attempt")
                        continue

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
                        # Handle audience participation using secure client ID
                        stored_client_id = self._client_ids.get(ws_id, secrets.token_urlsafe(16))
                        loop_id = data.get("loop_id", "")

                        # Validate loop_id exists and is active
                        if not loop_id or loop_id not in self.active_loops:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": f"Invalid or inactive loop_id: {loop_id}"}
                            }))
                            continue

                        # Validate payload structure and size (DoS protection)
                        payload = data.get("payload", {})
                        if not isinstance(payload, dict):
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Invalid payload format"}
                            }))
                            continue
                        # Limit payload size to 10KB
                        payload_str = json.dumps(payload)
                        if len(payload_str) > 10240:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Payload too large (max 10KB)"}
                            }))
                            continue

                        # Get or create rate limiter for this client
                        if stored_client_id not in self._rate_limiters:
                            self._rate_limiters[stored_client_id] = TokenBucket(
                                rate_per_minute=10.0,  # 10 messages per minute
                                burst_size=5  # Allow burst of 5
                            )

                        # Track access time for TTL-based cleanup
                        self._rate_limiter_last_access[stored_client_id] = time.time()

                        # Periodic cleanup to prevent memory leak
                        self._rate_limiter_cleanup_counter += 1
                        if self._rate_limiter_cleanup_counter >= self._CLEANUP_INTERVAL:
                            self._rate_limiter_cleanup_counter = 0
                            self._cleanup_stale_rate_limiters()

                        # Check rate limit
                        if not self._rate_limiters[stored_client_id].consume(1):
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Rate limited. Please wait before submitting again."}
                            }))
                            continue

                        # Create and queue the message
                        audience_msg = AudienceMessage(
                            type="vote" if msg_type == "user_vote" else "suggestion",
                            loop_id=loop_id,
                            payload=payload,
                            user_id=stored_client_id,
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

                except json.JSONDecodeError as e:
                    logger.warning(f"[ws] Invalid JSON from client: {e}")
        except Exception as e:
            # Connection closed errors are normal during shutdown
            error_name = type(e).__name__
            if "ConnectionClosed" in error_name or "ConnectionClosedOK" in error_name:
                logger.debug(f"[ws] Client {client_id[:8]}... disconnected normally")
            else:
                # Log unexpected errors for debugging (but don't expose to client)
                logger.error(f"[ws] Unexpected error for client {client_id[:8]}...: {error_name}: {e}")
        finally:
            self.clients.discard(websocket)
            # Clean up secure client ID mapping
            self._client_ids.pop(ws_id, None)

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
                except Exception as e:
                    logger.debug(f"Error closing WebSocket client: {e}")
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


# =============================================================================
# Unified HTTP + WebSocket Server (aiohttp-based)
# =============================================================================

class AiohttpUnifiedServer:
    """
    Unified server using aiohttp to handle both HTTP API and WebSocket on a single port.

    This is the recommended server for production as it avoids CORS issues with
    separate ports for HTTP and WebSocket.

    Usage:
        server = AiohttpUnifiedServer(port=8080, nomic_dir=Path(".nomic"))
        await server.start()
    """

    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        nomic_dir: Optional[Path] = None,
    ):
        self.port = port
        self.host = host
        self.nomic_dir = nomic_dir

        # WebSocket clients and event emitter
        self.clients: set = set()
        self._emitter = SyncEventEmitter()
        self._running = False

        # Multi-loop tracking
        self.active_loops: dict[str, LoopInstance] = {}
        self._active_loops_lock = threading.Lock()

        # Debate state caching
        self.debate_states: dict[str, dict] = {}
        self._debate_states_lock = threading.Lock()

        # Audience participation
        self.audience_inbox = AudienceInbox()
        self._rate_limiters: dict[str, TokenBucket] = {}
        self._rate_limiter_last_access: dict[str, float] = {}
        self._rate_limiters_lock = threading.Lock()

        # Secure client ID mapping
        self._client_ids: Dict[int, str] = {}

        # Optional stores (initialized from nomic_dir)
        self.elo_system = None
        self.insight_store = None
        self.flip_detector = None
        self.persona_manager = None
        self.debate_embeddings = None

        # ArgumentCartographer registry (loop_id -> cartographer instance)
        self.cartographers: Dict[str, Any] = {}
        self._cartographers_lock = threading.Lock()

        # Subscribe to emitter to maintain debate states
        self._emitter.subscribe(self._update_debate_state)

        # Initialize stores from nomic_dir
        if nomic_dir:
            self._init_stores(nomic_dir)

    def _init_stores(self, nomic_dir: Path) -> None:
        """Initialize optional stores from nomic directory."""
        # EloSystem for leaderboard
        try:
            from aragora.ranking.elo import EloSystem
            elo_path = nomic_dir / "agent_elo.db"
            if elo_path.exists():
                self.elo_system = EloSystem(str(elo_path))
                logger.info("[server] EloSystem loaded")
        except ImportError:
            logger.debug("[server] EloSystem not available (optional dependency)")

        # InsightStore for insights
        try:
            from aragora.insights.store import InsightStore
            insights_path = nomic_dir / "aragora_insights.db"
            if insights_path.exists():
                self.insight_store = InsightStore(str(insights_path))
                logger.info("[server] InsightStore loaded")
        except ImportError:
            logger.debug("[server] InsightStore not available (optional dependency)")

        # FlipDetector for position reversals
        try:
            from aragora.insights.flip_detector import FlipDetector
            positions_path = nomic_dir / "aragora_positions.db"
            if positions_path.exists():
                self.flip_detector = FlipDetector(str(positions_path))
                logger.info("[server] FlipDetector loaded")
        except ImportError:
            logger.debug("[server] FlipDetector not available (optional dependency)")

        # PersonaManager for agent specialization
        try:
            from aragora.personas.manager import PersonaManager
            personas_path = nomic_dir / "personas.db"
            if personas_path.exists():
                self.persona_manager = PersonaManager(str(personas_path))
                logger.info("[server] PersonaManager loaded")
        except ImportError:
            logger.debug("[server] PersonaManager not available (optional dependency)")

        # DebateEmbeddingsDatabase for memory
        try:
            from aragora.debate.embeddings import DebateEmbeddingsDatabase
            embeddings_path = nomic_dir / "debate_embeddings.db"
            if embeddings_path.exists():
                self.debate_embeddings = DebateEmbeddingsDatabase(str(embeddings_path))
                logger.info("[server] DebateEmbeddings loaded")
        except ImportError:
            logger.debug("[server] DebateEmbeddings not available (optional dependency)")

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter for nomic loop integration."""
        return self._emitter

    def _update_debate_state(self, event: StreamEvent) -> None:
        """Update cached debate state based on emitted events."""
        loop_id = event.loop_id
        with self._debate_states_lock:
            if event.type == StreamEventType.DEBATE_START:
                self.debate_states[loop_id] = {
                    "id": loop_id,
                    "task": event.data.get("task"),
                    "agents": event.data.get("agents"),
                    "started_at": event.timestamp,
                }

    def register_loop(self, loop_id: str, name: str, path: str = "") -> None:
        """Register a new nomic loop instance."""
        with self._active_loops_lock:
            self.active_loops[loop_id] = LoopInstance(
                loop_id=loop_id,
                name=name,
                started_at=time.time(),
                path=path,
            )
        # Broadcast loop registration
        self._emitter.emit(StreamEvent(
            type=StreamEventType.LOOP_REGISTER,
            data={"loop_id": loop_id, "name": name, "started_at": time.time(), "path": path},
            loop_id=loop_id,
        ))

    def unregister_loop(self, loop_id: str) -> None:
        """Unregister a nomic loop instance."""
        with self._active_loops_lock:
            self.active_loops.pop(loop_id, None)
        # Broadcast loop unregistration
        self._emitter.emit(StreamEvent(
            type=StreamEventType.LOOP_UNREGISTER,
            data={"loop_id": loop_id},
            loop_id=loop_id,
        ))

    def update_loop_state(self, loop_id: str, cycle: Optional[int] = None, phase: Optional[str] = None) -> None:
        """Update loop state (cycle/phase)."""
        with self._active_loops_lock:
            if loop_id in self.active_loops:
                if cycle is not None:
                    self.active_loops[loop_id].cycle = cycle
                if phase is not None:
                    self.active_loops[loop_id].phase = phase

    def register_cartographer(self, loop_id: str, cartographer: Any) -> None:
        """Register an ArgumentCartographer instance for a loop."""
        with self._cartographers_lock:
            self.cartographers[loop_id] = cartographer

    def unregister_cartographer(self, loop_id: str) -> None:
        """Unregister an ArgumentCartographer instance."""
        with self._cartographers_lock:
            self.cartographers.pop(loop_id, None)

    def _cors_headers(self, origin: Optional[str] = None) -> dict:
        """Generate CORS headers with proper origin validation.

        Only allows origins in the whitelist. Does NOT fallback to first
        origin for unauthorized requests (that would be a security issue).
        """
        headers = {
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400",
        }
        # Only add Allow-Origin for whitelisted origins or same-origin requests
        if origin and origin in WS_ALLOWED_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
        elif not origin:
            # Same-origin request - allow with wildcard
            headers["Access-Control-Allow-Origin"] = "*"
        # For unauthorized origins, don't add Allow-Origin (browser will block)
        return headers

    async def _handle_options(self, request) -> 'aiohttp.web.Response':
        """Handle CORS preflight requests."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")
        return web.Response(status=204, headers=self._cors_headers(origin))

    async def _handle_leaderboard(self, request) -> 'aiohttp.web.Response':
        """GET /api/leaderboard - Agent rankings."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.elo_system:
            return web.json_response(
                {"agents": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            limit = int(request.query.get("limit", 10))
            agents = self.elo_system.get_leaderboard(limit=limit)
            # Convert AgentRating objects to dicts
            agent_data = [
                {
                    "name": a.agent_name,
                    "elo": round(a.elo),
                    "wins": a.wins,
                    "losses": a.losses,
                    "draws": a.draws,
                    "win_rate": round(a.win_rate * 100, 1),
                    "games": a.games_played,
                }
                for a in agents
            ]
            return web.json_response(
                {"agents": agent_data, "count": len(agent_data)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Leaderboard error: {e}")
            return web.json_response(
                {"error": "Failed to fetch leaderboard"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_matches_recent(self, request) -> 'aiohttp.web.Response':
        """GET /api/matches/recent - Recent ELO matches."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.elo_system:
            return web.json_response(
                {"matches": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            limit = int(request.query.get("limit", 10))
            matches = self.elo_system.get_recent_matches(limit=limit)
            return web.json_response(
                {"matches": matches, "count": len(matches)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Matches error: {e}")
            return web.json_response(
                {"error": "Failed to fetch matches"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_insights_recent(self, request) -> 'aiohttp.web.Response':
        """GET /api/insights/recent - Recent debate insights."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.insight_store:
            return web.json_response(
                {"insights": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            limit = int(request.query.get("limit", 10))
            insights = self.insight_store.get_recent_insights(limit=limit)
            return web.json_response(
                {"insights": insights, "count": len(insights)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Insights error: {e}")
            return web.json_response(
                {"error": "Failed to fetch insights"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_flips_summary(self, request) -> 'aiohttp.web.Response':
        """GET /api/flips/summary - Position flip summary."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.flip_detector:
            return web.json_response(
                {"summary": {}, "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            summary = self.flip_detector.get_summary()
            return web.json_response(
                {"summary": summary, "count": summary.get("total_flips", 0)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Flips summary error: {e}")
            return web.json_response(
                {"error": "Failed to fetch flip summary"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_flips_recent(self, request) -> 'aiohttp.web.Response':
        """GET /api/flips/recent - Recent position flips."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.flip_detector:
            return web.json_response(
                {"flips": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            limit = int(request.query.get("limit", 10))
            flips = self.flip_detector.get_recent_flips(limit=limit)
            return web.json_response(
                {"flips": flips, "count": len(flips)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Flips recent error: {e}")
            return web.json_response(
                {"error": "Failed to fetch recent flips"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_tournaments(self, request) -> 'aiohttp.web.Response':
        """GET /api/tournaments - Tournament list with real data."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.nomic_dir:
            return web.json_response(
                {"tournaments": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            tournaments_dir = self.nomic_dir / "tournaments"
            tournaments_list = []

            if tournaments_dir.exists():
                for db_file in sorted(tournaments_dir.glob("*.db")):
                    try:
                        from aragora.tournaments.tournament import TournamentManager
                        manager = TournamentManager(db_path=str(db_file))

                        # Get tournament metadata
                        tournament = manager.get_tournament()
                        standings = manager.get_current_standings()
                        match_summary = manager.get_match_summary()

                        if tournament:
                            tournament["participants"] = len(standings)
                            tournament["total_matches"] = match_summary["total_matches"]
                            tournament["top_agent"] = standings[0].agent_name if standings else None
                            tournaments_list.append(tournament)
                    except Exception:
                        continue  # Skip corrupted files

            return web.json_response(
                {"tournaments": tournaments_list, "count": len(tournaments_list)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Tournament list error: {e}")
            return web.json_response(
                {"error": "Failed to fetch tournaments", "tournaments": [], "count": 0},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_tournament_details(self, request) -> 'aiohttp.web.Response':
        """GET /api/tournaments/{tournament_id} - Tournament details with standings."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        # Extract tournament_id from URL
        tournament_id = request.match_info.get('tournament_id', '')

        # Validate tournament_id format (prevent path traversal)
        if not re.match(r'^[a-zA-Z0-9_-]+$', tournament_id):
            return web.json_response(
                {"error": "Invalid tournament ID format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        if not self.nomic_dir:
            return web.json_response(
                {"error": "Nomic directory not configured"},
                status=503,
                headers=self._cors_headers(origin)
            )

        try:
            tournament_db = self.nomic_dir / "tournaments" / f"{tournament_id}.db"

            if not tournament_db.exists():
                return web.json_response(
                    {"error": "Tournament not found"},
                    status=404,
                    headers=self._cors_headers(origin)
                )

            from aragora.tournaments.tournament import TournamentManager
            manager = TournamentManager(db_path=str(tournament_db))

            tournament = manager.get_tournament()
            standings = manager.get_current_standings()
            matches = manager.get_matches(limit=100)

            if not tournament:
                return web.json_response(
                    {"error": "Tournament data not found"},
                    status=404,
                    headers=self._cors_headers(origin)
                )

            # Format standings for API response
            standings_data = [
                {
                    "agent": s.agent_name,
                    "wins": s.wins,
                    "losses": s.losses,
                    "draws": s.draws,
                    "points": s.points,
                    "total_score": round(s.total_score, 2),
                    "matches_played": s.matches_played,
                    "win_rate": round(s.win_rate * 100, 1),
                }
                for s in standings
            ]

            return web.json_response(
                {
                    "tournament": tournament,
                    "standings": standings_data,
                    "standings_count": len(standings_data),
                    "recent_matches": matches,
                    "matches_count": len(matches),
                },
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Tournament details error: {e}")
            return web.json_response(
                {"error": "Failed to fetch tournament details"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_agent_consistency(self, request) -> 'aiohttp.web.Response':
        """GET /api/agent/{name}/consistency - Agent consistency score from FlipDetector."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        agent_name = request.match_info.get('name', '')

        # Validate agent name format
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_name):
            return web.json_response(
                {"error": "Invalid agent name format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        try:
            from aragora.insights.flip_detector import FlipDetector

            db_path = self.nomic_dir / "aragora_personas.db" if self.nomic_dir else "aragora_personas.db"
            detector = FlipDetector(db_path=str(db_path))

            # Get consistency score
            score = detector.get_agent_consistency(agent_name)

            if score:
                consistency = score.consistency_score
                consistency_class = "high" if consistency >= 0.8 else ("medium" if consistency >= 0.5 else "low")
                return web.json_response(
                    {
                        "agent": agent_name,
                        "consistency": consistency,
                        "consistency_class": consistency_class,
                        "total_positions": score.total_positions,
                        "total_flips": score.total_flips,
                        "flip_rate": score.flip_rate,
                        "contradictions": score.contradictions,
                        "refinements": score.refinements,
                    },
                    headers=self._cors_headers(origin)
                )
            else:
                # No data yet - return default high consistency
                return web.json_response(
                    {
                        "agent": agent_name,
                        "consistency": 1.0,
                        "consistency_class": "high",
                        "total_positions": 0,
                        "total_flips": 0,
                        "flip_rate": 0.0,
                        "contradictions": 0,
                        "refinements": 0,
                    },
                    headers=self._cors_headers(origin)
                )
        except Exception as e:
            logger.error(f"Agent consistency error for {agent_name}: {e}")
            return web.json_response(
                {"error": "Failed to fetch agent consistency"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_agent_network(self, request) -> 'aiohttp.web.Response':
        """GET /api/agent/{name}/network - Agent relationship network (rivals, allies)."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        agent_name = request.match_info.get('name', '')

        # Validate agent name format
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_name):
            return web.json_response(
                {"error": "Invalid agent name format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        try:
            # Try to get relationship data from ELO system or persona manager
            network_data = {
                "agent": agent_name,
                "influences": [],
                "influenced_by": [],
                "rivals": [],
                "allies": [],
            }

            # Try persona manager first (has relationship tracker)
            if self.persona_manager and hasattr(self.persona_manager, 'relationship_tracker'):
                tracker = self.persona_manager.relationship_tracker

                if hasattr(tracker, 'get_rivals'):
                    rivals = tracker.get_rivals(agent_name, limit=5)
                    network_data["rivals"] = [
                        {"agent": r[0], "score": r[1], "debate_count": 0}
                        for r in rivals
                    ] if rivals else []

                if hasattr(tracker, 'get_allies'):
                    allies = tracker.get_allies(agent_name, limit=5)
                    network_data["allies"] = [
                        {"agent": a[0], "score": a[1], "debate_count": 0}
                        for a in allies
                    ] if allies else []

                if hasattr(tracker, 'get_influence_network'):
                    influence = tracker.get_influence_network(agent_name)
                    network_data["influences"] = [
                        {"agent": name, "score": score, "debate_count": 0}
                        for name, score in influence.get("influences", [])
                    ]
                    network_data["influenced_by"] = [
                        {"agent": name, "score": score, "debate_count": 0}
                        for name, score in influence.get("influenced_by", [])
                    ]

            # Fall back to ELO system if no persona manager
            elif self.elo_system:
                if hasattr(self.elo_system, 'get_rivals'):
                    rivals = self.elo_system.get_rivals(agent_name, limit=5)
                    network_data["rivals"] = [
                        {"agent": r.get("agent_b", r.get("agent")), "score": r.get("rivalry_score", 0), "debate_count": r.get("matches", 0)}
                        for r in rivals
                    ] if rivals else []

                if hasattr(self.elo_system, 'get_allies'):
                    allies = self.elo_system.get_allies(agent_name, limit=5)
                    network_data["allies"] = [
                        {"agent": a.get("agent_b", a.get("agent")), "score": a.get("alliance_score", 0), "debate_count": a.get("matches", 0)}
                        for a in allies
                    ] if allies else []

            return web.json_response(network_data, headers=self._cors_headers(origin))
        except Exception as e:
            logger.error(f"Agent network error for {agent_name}: {e}")
            return web.json_response(
                {"error": "Failed to fetch agent network"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_memory_tier_stats(self, request) -> 'aiohttp.web.Response':
        """GET /api/memory/tier-stats - Continuum memory statistics."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.debate_embeddings:
            return web.json_response(
                {"tiers": {"fast": 0, "medium": 0, "slow": 0, "glacial": 0}, "total": 0},
                headers=self._cors_headers(origin)
            )

        try:
            stats = self.debate_embeddings.get_tier_stats() if hasattr(self.debate_embeddings, 'get_tier_stats') else {}
            return web.json_response(
                {"tiers": stats, "total": sum(stats.values()) if stats else 0},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Memory tier stats error: {e}")
            return web.json_response(
                {"error": "Failed to fetch memory stats"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_laboratory_emergent_traits(self, request) -> 'aiohttp.web.Response':
        """GET /api/laboratory/emergent-traits - Discovered agent traits."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.persona_manager:
            return web.json_response(
                {"traits": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            min_confidence = float(request.query.get("min_confidence", 0.3))
            limit = int(request.query.get("limit", 10))
            traits = self.persona_manager.get_emergent_traits(min_confidence=min_confidence, limit=limit) if hasattr(self.persona_manager, 'get_emergent_traits') else []
            return web.json_response(
                {"traits": traits, "count": len(traits)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Emergent traits error: {e}")
            return web.json_response(
                {"error": "Failed to fetch emergent traits"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_laboratory_cross_pollinations(self, request) -> 'aiohttp.web.Response':
        """GET /api/laboratory/cross-pollinations/suggest - Trait transfer suggestions."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.persona_manager:
            return web.json_response(
                {"suggestions": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            suggestions = self.persona_manager.suggest_cross_pollinations() if hasattr(self.persona_manager, 'suggest_cross_pollinations') else []
            return web.json_response(
                {"suggestions": suggestions, "count": len(suggestions)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Cross-pollinations error: {e}")
            return web.json_response(
                {"error": "Failed to fetch cross-pollination suggestions"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_nomic_state(self, request) -> 'aiohttp.web.Response':
        """GET /api/nomic/state - Current nomic loop state."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        with self._active_loops_lock:
            if self.active_loops:
                loop = list(self.active_loops.values())[0]
                state = {
                    "cycle": loop.cycle,
                    "phase": loop.phase,
                    "loop_id": loop.loop_id,
                    "name": loop.name,
                }
            else:
                state = {"cycle": 0, "phase": "idle"}

        return web.json_response(state, headers=self._cors_headers(origin))

    async def _handle_graph_json(self, request) -> 'aiohttp.web.Response':
        """GET /api/debate/{loop_id}/graph - Debate argument graph as JSON."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        loop_id = request.match_info.get('loop_id', '')

        # Validate loop_id format (security: prevent injection)
        if not re.match(r'^[a-zA-Z0-9_-]+$', loop_id):
            return web.json_response(
                {"error": "Invalid loop_id format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        with self._cartographers_lock:
            cartographer = self.cartographers.get(loop_id)

        if not cartographer:
            return web.json_response(
                {"error": f"No cartographer found for loop: {loop_id}"},
                status=404,
                headers=self._cors_headers(origin)
            )

        try:
            include_full = request.query.get("full", "false").lower() == "true"
            graph_json = cartographer.export_json(include_full_content=include_full)
            return web.Response(
                text=graph_json,
                content_type="application/json",
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Graph JSON error for {loop_id}: {e}")
            return web.json_response(
                {"error": "Failed to export graph"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_graph_mermaid(self, request) -> 'aiohttp.web.Response':
        """GET /api/debate/{loop_id}/graph/mermaid - Debate argument graph as Mermaid diagram."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        loop_id = request.match_info.get('loop_id', '')

        # Validate loop_id format
        if not re.match(r'^[a-zA-Z0-9_-]+$', loop_id):
            return web.json_response(
                {"error": "Invalid loop_id format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        with self._cartographers_lock:
            cartographer = self.cartographers.get(loop_id)

        if not cartographer:
            return web.json_response(
                {"error": f"No cartographer found for loop: {loop_id}"},
                status=404,
                headers=self._cors_headers(origin)
            )

        try:
            direction = request.query.get("direction", "TD")
            # Validate direction (only TD or LR)
            if direction not in ("TD", "LR"):
                direction = "TD"
            mermaid = cartographer.export_mermaid(direction=direction)
            return web.Response(
                text=mermaid,
                content_type="text/plain",
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Graph Mermaid error for {loop_id}: {e}")
            return web.json_response(
                {"error": "Failed to export Mermaid diagram"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_graph_stats(self, request) -> 'aiohttp.web.Response':
        """GET /api/debate/{loop_id}/graph/stats - Debate argument graph statistics."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        loop_id = request.match_info.get('loop_id', '')

        # Validate loop_id format
        if not re.match(r'^[a-zA-Z0-9_-]+$', loop_id):
            return web.json_response(
                {"error": "Invalid loop_id format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        with self._cartographers_lock:
            cartographer = self.cartographers.get(loop_id)

        if not cartographer:
            return web.json_response(
                {"error": f"No cartographer found for loop: {loop_id}"},
                status=404,
                headers=self._cors_headers(origin)
            )

        try:
            stats = cartographer.get_statistics()
            return web.json_response(stats, headers=self._cors_headers(origin))
        except Exception as e:
            logger.error(f"Graph stats error for {loop_id}: {e}")
            return web.json_response(
                {"error": "Failed to get graph statistics"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_audience_clusters(self, request) -> 'aiohttp.web.Response':
        """GET /api/debate/{loop_id}/audience/clusters - Clustered audience suggestions."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        loop_id = request.match_info.get('loop_id', '')

        # Validate loop_id format
        if not re.match(r'^[a-zA-Z0-9_-]+$', loop_id):
            return web.json_response(
                {"error": "Invalid loop_id format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        try:
            from aragora.audience.suggestions import cluster_suggestions

            # Get suggestions from audience inbox for this loop
            suggestions = self.audience_inbox.drain_suggestions(loop_id=loop_id)

            if not suggestions:
                return web.json_response(
                    {"clusters": [], "total": 0},
                    headers=self._cors_headers(origin)
                )

            # Cluster suggestions
            clusters = cluster_suggestions(
                suggestions,
                similarity_threshold=float(request.query.get("threshold", 0.6)),
                max_clusters=int(request.query.get("max_clusters", 5)),
            )

            return web.json_response(
                {
                    "clusters": [
                        {
                            "representative": c.representative,
                            "count": c.count,
                            "user_ids": c.user_ids[:5],  # Limit user IDs for privacy
                        }
                        for c in clusters
                    ],
                    "total": sum(c.count for c in clusters),
                },
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Audience clusters error for {loop_id}: {e}")
            return web.json_response(
                {"error": "Failed to cluster audience suggestions"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_replays(self, request) -> 'aiohttp.web.Response':
        """GET /api/replays - List available debate replays."""
        import aiohttp.web as web
        origin = request.headers.get("Origin")

        if not self.nomic_dir:
            return web.json_response(
                {"replays": [], "count": 0},
                headers=self._cors_headers(origin)
            )

        try:
            replays_dir = self.nomic_dir / "replays"
            if not replays_dir.exists():
                return web.json_response(
                    {"replays": [], "count": 0},
                    headers=self._cors_headers(origin)
                )

            replays = []
            for replay_path in replays_dir.iterdir():
                if replay_path.is_dir():
                    meta_file = replay_path / "meta.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text())
                        replays.append({
                            "id": replay_path.name,
                            "topic": meta.get("topic", replay_path.name),
                            "timestamp": meta.get("timestamp", ""),
                        })

            return web.json_response(
                {"replays": sorted(replays, key=lambda x: x["id"], reverse=True), "count": len(replays)},
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Replays list error: {e}")
            return web.json_response(
                {"error": "Failed to list replays"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_replay_html(self, request) -> 'aiohttp.web.Response':
        """GET /api/replays/{replay_id}/html - Get HTML replay visualization."""
        import aiohttp.web as web
        import re
        origin = request.headers.get("Origin")

        replay_id = request.match_info.get('replay_id', '')

        # Validate replay_id format (security: prevent path traversal)
        if not re.match(r'^[a-zA-Z0-9_-]+$', replay_id):
            return web.json_response(
                {"error": "Invalid replay_id format"},
                status=400,
                headers=self._cors_headers(origin)
            )

        if not self.nomic_dir:
            return web.json_response(
                {"error": "No nomic directory configured"},
                status=500,
                headers=self._cors_headers(origin)
            )

        try:
            replay_dir = self.nomic_dir / "replays" / replay_id
            if not replay_dir.exists():
                return web.json_response(
                    {"error": f"Replay not found: {replay_id}"},
                    status=404,
                    headers=self._cors_headers(origin)
                )

            # Check for pre-generated HTML
            html_file = replay_dir / "replay.html"
            if html_file.exists():
                return web.Response(
                    text=html_file.read_text(),
                    content_type="text/html",
                    headers=self._cors_headers(origin)
                )

            # Generate from events.jsonl if no pre-generated HTML
            events_file = replay_dir / "events.jsonl"
            meta_file = replay_dir / "meta.json"

            if not events_file.exists():
                return web.json_response(
                    {"error": f"No events found for replay: {replay_id}"},
                    status=404,
                    headers=self._cors_headers(origin)
                )

            # Load events and generate HTML
            from aragora.visualization.replay import ReplayGenerator, ReplayArtifact, ReplayScene
            from aragora.core import Message
            from datetime import datetime

            events = []
            with events_file.open() as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))

            # Create artifact from events
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            generator = ReplayGenerator()

            # Simple HTML generation from events
            artifact = ReplayArtifact(
                debate_id=replay_id,
                task=meta.get("topic", "Unknown"),
                scenes=[],
                verdict=meta.get("verdict", {}),
                metadata=meta,
            )

            # Group events by round
            round_events: Dict[int, list] = {}
            for event in events:
                round_num = event.get("round", 0)
                round_events.setdefault(round_num, []).append(event)

            for round_num in sorted(round_events.keys()):
                messages = []
                for event in round_events[round_num]:
                    if event.get("type") in ("agent_message", "propose", "critique"):
                        messages.append(Message(
                            role=event.get("data", {}).get("role", "unknown"),
                            agent=event.get("agent", "unknown"),
                            content=event.get("data", {}).get("content", ""),
                            round=round_num,
                        ))
                if messages:
                    artifact.scenes.append(ReplayScene(
                        round_number=round_num,
                        timestamp=datetime.now(),
                        messages=messages,
                    ))

            html = generator._render_html(artifact)
            return web.Response(
                text=html,
                content_type="text/html",
                headers=self._cors_headers(origin)
            )
        except Exception as e:
            logger.error(f"Replay HTML error for {replay_id}: {e}")
            return web.json_response(
                {"error": "Failed to generate replay HTML"},
                status=500,
                headers=self._cors_headers(origin)
            )

    async def _handle_start_debate(self, request) -> 'aiohttp.web.Response':
        """POST /api/debate - Start an ad-hoc debate with specified question.

        Accepts JSON body with:
            question: The topic/question to debate (required)
            agents: Comma-separated agent list (optional, default: "anthropic-api,openai-api,gemini,grok")
            rounds: Number of debate rounds (optional, default: 3)
            consensus: Consensus method (optional, default: "majority")

        All agents participate as proposers for full participation in all rounds.
        """
        global _active_debates, _debate_executor
        import aiohttp.web as web

        origin = request.headers.get("Origin")

        if not DEBATE_AVAILABLE:
            return web.json_response(
                {"error": "Debate orchestrator not available"},
                status=500,
                headers=self._cors_headers(origin)
            )

        # Parse JSON body
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400,
                headers=self._cors_headers(origin)
            )

        # Validate required fields with length limits
        question = data.get('question', '').strip()
        if not question:
            return web.json_response(
                {"error": "question field is required"},
                status=400,
                headers=self._cors_headers(origin)
            )
        if len(question) > 10000:
            return web.json_response(
                {"error": "question must be under 10,000 characters"},
                status=400,
                headers=self._cors_headers(origin)
            )

        # Parse optional fields with validation
        agents_str = data.get('agents', 'anthropic-api,openai-api,gemini,grok')
        try:
            rounds = min(max(int(data.get('rounds', 3)), 1), 10)  # Clamp to 1-10
        except (ValueError, TypeError):
            rounds = 3
        consensus = data.get('consensus', 'majority')

        # Generate debate ID
        debate_id = f"adhoc_{uuid.uuid4().hex[:8]}"

        # Track this debate (thread-safe)
        with _active_debates_lock:
            _active_debates[debate_id] = {
                "id": debate_id,
                "question": question,
                "status": "starting",
                "agents": agents_str,
                "rounds": rounds,
            }

        # Set loop_id on emitter so events are tagged
        self.emitter.set_loop_id(debate_id)

        # Start debate in background thread
        def run_debate():
            import asyncio as _asyncio

            try:
                # Parse agents with bounds check
                agent_list = [s.strip() for s in agents_str.split(",") if s.strip()]
                if len(agent_list) > MAX_AGENTS_PER_DEBATE:
                    with _active_debates_lock:
                        _active_debates[debate_id]["status"] = "error"
                        _active_debates[debate_id]["error"] = f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}"
                    return
                if len(agent_list) < 2:
                    with _active_debates_lock:
                        _active_debates[debate_id]["status"] = "error"
                        _active_debates[debate_id]["error"] = "At least 2 agents required for a debate"
                    return

                agent_specs = []
                for spec in agent_list:
                    spec = spec.strip()
                    if ":" in spec:
                        agent_type, role = spec.split(":", 1)
                    else:
                        agent_type = spec
                        role = None
                    # Validate agent type against allowlist
                    if agent_type.lower() not in ALLOWED_AGENT_TYPES:
                        raise ValueError(f"Invalid agent type: {agent_type}. Allowed: {', '.join(sorted(ALLOWED_AGENT_TYPES))}")
                    agent_specs.append((agent_type, role))

                # Create agents with streaming support
                # All agents are proposers for full participation in all rounds
                agents = []
                for i, (agent_type, role) in enumerate(agent_specs):
                    if role is None:
                        role = "proposer"  # All agents propose and participate fully
                    agent = create_agent(
                        model_type=agent_type,
                        name=f"{agent_type}_{role}",
                        role=role,
                    )
                    # Wrap agent for token streaming if supported
                    agent = _wrap_agent_for_streaming(agent, self.emitter, debate_id)
                    agents.append(agent)

                # Create environment and protocol
                env = Environment(task=question, context="", max_rounds=rounds)
                protocol = DebateProtocol(
                    rounds=rounds,
                    consensus=consensus,
                    proposer_count=len(agents),  # All agents propose initially
                    topology="all-to-all",  # Everyone critiques everyone
                )

                # Create arena with hooks and available context systems
                hooks = create_arena_hooks(self.emitter)
                arena = Arena(
                    env, agents, protocol,
                    event_hooks=hooks,
                    event_emitter=self.emitter,
                    loop_id=debate_id,
                )

                # Run debate with timeout protection (10 minutes max)
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "running"

                async def run_with_timeout():
                    return await _asyncio.wait_for(arena.run(), timeout=600)

                result = _asyncio.run(run_with_timeout())
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "completed"
                    _active_debates[debate_id]["result"] = {
                        "final_answer": result.final_answer,
                        "consensus_reached": result.consensus_reached,
                        "confidence": result.confidence,
                    }

            except Exception as e:
                import traceback
                safe_msg = _safe_error_message(e, "debate_execution")
                error_trace = traceback.format_exc()
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "error"
                    _active_debates[debate_id]["error"] = safe_msg
                logger.error(f"[debate] Thread error in {debate_id}: {str(e)}\n{error_trace}")
                # Emit error event to client
                self.emitter.emit(StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"error": safe_msg, "debate_id": debate_id},
                ))

        # Use thread pool to prevent unbounded thread creation
        if _debate_executor is None:
            with _debate_executor_lock:
                if _debate_executor is None:
                    _debate_executor = ThreadPoolExecutor(
                        max_workers=MAX_CONCURRENT_DEBATES,
                        thread_name_prefix="debate-"
                    )

        try:
            _debate_executor.submit(run_debate)
        except RuntimeError:
            return web.json_response({
                "success": False,
                "error": "Server at capacity. Please try again later.",
            }, status=503, headers=self._cors_headers(origin))

        # Return immediately with debate ID
        return web.json_response({
            "success": True,
            "debate_id": debate_id,
            "question": question,
            "agents": agents_str.split(","),
            "rounds": rounds,
            "status": "starting",
            "message": "Debate started. Connect to WebSocket to receive events.",
        }, headers=self._cors_headers(origin))

    async def _websocket_handler(self, request) -> 'aiohttp.web.WebSocketResponse':
        """Handle WebSocket connections with security validation and optional auth."""
        import aiohttp
        import aiohttp.web as web

        # Validate origin for security (match websockets handler behavior)
        origin = request.headers.get("Origin", "")
        if origin and origin not in WS_ALLOWED_ORIGINS:
            # Reject connection from unauthorized origin
            return web.Response(status=403, text="Origin not allowed")

        # Optional authentication (controlled by ARAGORA_API_TOKEN env var)
        try:
            from aragora.server.auth import auth_config, check_auth

            if auth_config.enabled:
                # Convert headers to dict for check_auth
                headers = dict(request.headers)
                query_string = request.url.query_string or ""

                # Get client IP (validate proxy headers for security)
                remote_ip = request.remote or ""
                if remote_ip in TRUSTED_PROXIES:
                    # Only trust X-Forwarded-For from trusted proxies
                    forwarded = request.headers.get('X-Forwarded-For', '')
                    client_ip = forwarded.split(',')[0].strip() if forwarded else remote_ip
                else:
                    # Untrusted source - use direct connection IP
                    client_ip = remote_ip

                authenticated, remaining = check_auth(
                    headers, query_string, loop_id="", ip_address=client_ip
                )

                if not authenticated:
                    status = 429 if remaining == 0 else 401
                    msg = "Rate limit exceeded" if remaining == 0 else "Authentication required"
                    return web.Response(status=status, text=msg)
        except ImportError:
            # Log warning if auth is required but module unavailable
            if os.getenv('ARAGORA_AUTH_REQUIRED'):
                logger.warning("[ws] Auth required but module unavailable - rejecting connection")
                return web.Response(status=500, text="Authentication system unavailable")
            pass  # Auth module not available and not required, continue

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.clients.add(ws)
        ws_id = id(ws)
        client_id = secrets.token_hex(16)
        self._client_ids[ws_id] = client_id

        # Initialize rate limiter for this client
        self._rate_limiters[client_id] = TokenBucket(
            rate_per_minute=10.0,  # 10 messages per minute
            burst_size=5  # Allow burst of 5
        )
        self._rate_limiter_last_access[client_id] = time.time()

        logger.info(f"[ws] Client connected: {client_id[:8]}...")

        # Send initial loop list
        with self._active_loops_lock:
            loops_data = [
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

        await ws.send_json({
            "type": "loop_list",
            "data": {"loops": loops_data, "count": len(loops_data)},
        })

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "get_loops":
                            with self._active_loops_lock:
                                loops_data = [
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
                            await ws.send_json({
                                "type": "loop_list",
                                "data": {"loops": loops_data, "count": len(loops_data)},
                            })

                        elif msg_type in ("user_vote", "user_suggestion"):
                            # Handle audience participation with validation
                            loop_id = data.get("loop_id", "")

                            # Validate loop_id exists and is active
                            with self._active_loops_lock:
                                loop_valid = loop_id and loop_id in self.active_loops

                            if not loop_valid:
                                await ws.send_json({
                                    "type": "error",
                                    "data": {"message": f"Invalid or inactive loop_id: {loop_id}"}
                                })
                                continue

                            # Validate payload structure and size (DoS protection)
                            payload = data.get("payload", {})
                            if not isinstance(payload, dict):
                                await ws.send_json({
                                    "type": "error",
                                    "data": {"message": "Invalid payload format"}
                                })
                                continue

                            # Limit payload size to 10KB
                            try:
                                payload_str = json.dumps(payload)
                                if len(payload_str) > 10240:
                                    await ws.send_json({
                                        "type": "error",
                                        "data": {"message": "Payload too large (max 10KB)"}
                                    })
                                    continue
                            except (TypeError, ValueError):
                                await ws.send_json({
                                    "type": "error",
                                    "data": {"message": "Invalid payload structure"}
                                })
                                continue

                            # Check rate limit
                            self._rate_limiter_last_access[client_id] = time.time()
                            if not self._rate_limiters[client_id].consume(1):
                                await ws.send_json({
                                    "type": "error",
                                    "data": {"message": "Rate limit exceeded, try again later"}
                                })
                                continue

                            audience_msg = AudienceMessage(
                                type="vote" if msg_type == "user_vote" else "suggestion",
                                loop_id=loop_id,
                                payload=payload,
                                user_id=client_id,
                            )
                            self.audience_inbox.put(audience_msg)

                            await ws.send_json({
                                "type": "ack",
                                "data": {"message": "Message received", "msg_type": msg_type}
                            })

                    except json.JSONDecodeError:
                        logger.warning(f"[ws] Invalid JSON from client")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'[ws] Error: {ws.exception()}')

        finally:
            self.clients.discard(ws)
            self._client_ids.pop(ws_id, None)
            # Clean up rate limiter for this client
            self._rate_limiters.pop(client_id, None)
            self._rate_limiter_last_access.pop(client_id, None)
            logger.info(f"[ws] Client disconnected: {client_id[:8]}...")

        return ws

    async def _drain_loop(self) -> None:
        """Drain events from the sync emitter and broadcast to WebSocket clients."""
        import aiohttp

        while self._running:
            try:
                event = self._emitter._queue.get(timeout=0.1)

                # Update loop state for cycle/phase events
                if event.type == StreamEventType.CYCLE_START:
                    self.update_loop_state(event.loop_id, cycle=event.data.get("cycle"))
                elif event.type == StreamEventType.PHASE_START:
                    self.update_loop_state(event.loop_id, phase=event.data.get("phase"))

                # Serialize event
                event_dict = {
                    "type": event.type.value,
                    "data": event.data,
                    "timestamp": event.timestamp,
                    "round": event.round,
                    "agent": event.agent,
                    "loop_id": event.loop_id,
                }
                message = json.dumps(event_dict)

                # Broadcast to all clients
                dead_clients = []
                for client in list(self.clients):
                    try:
                        await client.send_str(message)
                    except Exception:
                        dead_clients.append(client)

                for client in dead_clients:
                    self.clients.discard(client)

            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"[ws] Drain loop error: {e}")
                await asyncio.sleep(0.1)

    async def start(self) -> None:
        """Start the unified HTTP+WebSocket server."""
        import aiohttp.web as web

        self._running = True

        # Create aiohttp app
        app = web.Application()

        # Add routes
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)
        app.router.add_get("/api/leaderboard", self._handle_leaderboard)
        app.router.add_get("/api/matches/recent", self._handle_matches_recent)
        app.router.add_get("/api/insights/recent", self._handle_insights_recent)
        app.router.add_get("/api/flips/summary", self._handle_flips_summary)
        app.router.add_get("/api/flips/recent", self._handle_flips_recent)
        app.router.add_get("/api/tournaments", self._handle_tournaments)
        app.router.add_get("/api/tournaments/{tournament_id}", self._handle_tournament_details)
        app.router.add_get("/api/agent/{name}/consistency", self._handle_agent_consistency)
        app.router.add_get("/api/agent/{name}/network", self._handle_agent_network)
        app.router.add_get("/api/memory/tier-stats", self._handle_memory_tier_stats)
        app.router.add_get("/api/laboratory/emergent-traits", self._handle_laboratory_emergent_traits)
        app.router.add_get("/api/laboratory/cross-pollinations/suggest", self._handle_laboratory_cross_pollinations)
        app.router.add_get("/api/nomic/state", self._handle_nomic_state)
        app.router.add_get("/api/debate/{loop_id}/graph", self._handle_graph_json)
        app.router.add_get("/api/debate/{loop_id}/graph/mermaid", self._handle_graph_mermaid)
        app.router.add_get("/api/debate/{loop_id}/graph/stats", self._handle_graph_stats)
        app.router.add_get("/api/debate/{loop_id}/audience/clusters", self._handle_audience_clusters)
        app.router.add_get("/api/replays", self._handle_replays)
        app.router.add_get("/api/replays/{replay_id}/html", self._handle_replay_html)
        app.router.add_post("/api/debate", self._handle_start_debate)  # Start ad-hoc debate
        app.router.add_get("/", self._websocket_handler)  # WebSocket at root
        app.router.add_get("/ws", self._websocket_handler)  # Also at /ws

        # Start drain loop
        asyncio.create_task(self._drain_loop())

        # Run server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)

        print(f"Unified server (HTTP+WS) running on http://{self.host}:{self.port}")
        print(f"  WebSocket: ws://{self.host}:{self.port}/")
        print(f"  HTTP API:  http://{self.host}:{self.port}/api/*")

        await site.start()

        # Keep running
        try:
            await asyncio.Future()
        finally:
            self._running = False
            await runner.cleanup()

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
