"""
Real-time debate streaming via WebSocket.

The SyncEventEmitter bridges synchronous Arena code with async WebSocket broadcasts.
Events are queued synchronously and consumed by an async drain loop.

This module also supports unified HTTP+WebSocket serving on a single port via aiohttp.

Note: Core components are now in submodules for better organization:
- aragora.server.stream.events - StreamEventType, StreamEvent, AudienceMessage
- aragora.server.stream.emitter - SyncEventEmitter, TokenBucket, AudienceInbox
- aragora.server.stream.state_manager - DebateStateManager, BoundedDebateDict
- aragora.server.stream.arena_hooks - create_arena_hooks, wrap_agent_for_streaming
"""

import asyncio
import json
import logging
import os
import queue
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Any, Dict
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor
import uuid

# Configure module logger
logger = logging.getLogger(__name__)

# Import from sibling modules (core streaming components)
from .events import (
    StreamEventType,
    StreamEvent,
    AudienceMessage,
)
from .emitter import (
    TokenBucket,
    AudienceInbox,
    SyncEventEmitter,
    normalize_intensity,
)
from .state_manager import (
    BoundedDebateDict,
    LoopInstance,
    DebateStateManager,
    get_active_debates,
    get_active_debates_lock,
    get_debate_executor,
    set_debate_executor,
    get_debate_executor_lock,
    cleanup_stale_debates,
    increment_cleanup_counter,
)
from .arena_hooks import (
    create_arena_hooks,
    wrap_agent_for_streaming,
)
from .stream_handlers import StreamAPIHandlersMixin

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

# Import centralized config and error utilities
from aragora.config import (
    DB_INSIGHTS_PATH,
    DB_PERSONAS_PATH,
    MAX_AGENTS_PER_DEBATE,
    MAX_CONCURRENT_DEBATES,
    ALLOWED_AGENT_TYPES,
)
from aragora.server.error_utils import safe_error_message as _safe_error_message

# Backward compatibility aliases
_active_debates = get_active_debates()
_active_debates_lock = get_active_debates_lock()
_debate_executor_lock = get_debate_executor_lock()

# TTL for completed debates (24 hours)
_DEBATE_TTL_SECONDS = 86400


def _cleanup_stale_debates_stream() -> None:
    """Remove completed/errored debates older than TTL."""
    cleanup_stale_debates()


# Backward compatibility alias - use wrap_agent_for_streaming from arena_hooks
_wrap_agent_for_streaming = wrap_agent_for_streaming


# Centralized CORS configuration
from aragora.server.cors_config import WS_ALLOWED_ORIGINS

# Import WebSocket config from centralized location
from aragora.config import WS_MAX_MESSAGE_SIZE

# Import auth for WebSocket authentication
from aragora.server.auth import auth_config

# Trusted proxies for X-Forwarded-For header validation
# Only trust X-Forwarded-For if request comes from these IPs
TRUSTED_PROXIES = frozenset(
    p.strip() for p in os.getenv('ARAGORA_TRUSTED_PROXIES', '127.0.0.1,::1,localhost').split(',')
)


# =============================================================================
# NOTE: Core streaming classes are now in submodules for better organization:
# - StreamEventType, StreamEvent, AudienceMessage -> aragora.server.stream.events
# - TokenBucket, AudienceInbox, SyncEventEmitter -> aragora.server.stream.emitter
# - BoundedDebateDict, LoopInstance, DebateStateManager -> aragora.server.stream.state_manager
# - create_arena_hooks, wrap_agent_for_streaming -> aragora.server.stream.arena_hooks
#
# The classes are imported at the top of this file for backward compatibility.
# =============================================================================


class DebateStreamServer:
    """
    WebSocket server broadcasting debate events to connected clients.

    Supports multiple concurrent nomic loop instances with view switching.

    Lock Hierarchy (acquire in this order to prevent deadlocks):
    ---------------------------------------------------------------
    1. _rate_limiters_lock  - Protects _rate_limiters and _rate_limiter_last_access
    2. _debate_states_lock  - Protects debate_states and _debate_states_last_access
    3. _active_loops_lock   - Protects active_loops and _active_loops_last_access

    IMPORTANT: Each method should only acquire ONE lock at a time. If multiple
    locks must be acquired (e.g., in cleanup methods), acquire them sequentially
    in the order above, releasing each before acquiring the next. Never nest
    lock acquisitions to prevent deadlocks.

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
        # Audience participation - Lock hierarchy level 1 (acquire first)
        self.audience_inbox = AudienceInbox()
        self._rate_limiters: dict[str, TokenBucket] = {}  # client_id -> TokenBucket
        self._rate_limiter_last_access: dict[str, float] = {}  # client_id -> last access time
        self._rate_limiters_lock = threading.Lock()  # Lock #1 in hierarchy
        self._rate_limiter_cleanup_counter = 0  # Counter for periodic cleanup
        self._RATE_LIMITER_TTL = 3600  # 1 hour TTL for rate limiters
        self._CLEANUP_INTERVAL = 100  # Cleanup every N accesses

        # Debate state caching for late joiner sync - Lock hierarchy level 2
        self.debate_states: dict[str, dict] = {}  # loop_id -> debate state
        self._debate_states_lock = threading.Lock()  # Lock #2 in hierarchy
        self._debate_states_last_access: dict[str, float] = {}  # loop_id -> last access time
        self._DEBATE_STATES_TTL = 3600  # 1 hour TTL for ended debates
        self._MAX_DEBATE_STATES = 500  # Max cached states

        # Multi-loop tracking with thread safety - Lock hierarchy level 3 (acquire last)
        self.active_loops: dict[str, LoopInstance] = {}  # loop_id -> LoopInstance
        self._active_loops_lock = threading.Lock()  # Lock #3 in hierarchy
        self._active_loops_last_access: dict[str, float] = {}  # loop_id -> last access time
        self._ACTIVE_LOOPS_TTL = 86400  # 24 hour TTL for stale loops
        self._MAX_ACTIVE_LOOPS = 1000  # Max concurrent loops

        # Secure client ID mapping with LRU eviction (cryptographically random, not memory address)
        self._client_ids: OrderedDict[int, str] = OrderedDict()  # websocket id -> secure client_id
        self._MAX_CLIENT_IDS = 10000  # Max tracked clients

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
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limiters")

    def _extract_ws_token(self, websocket) -> Optional[str]:
        """Extract authentication token from WebSocket connection.

        Attempts to extract token from Authorization header only.
        Query parameter tokens are not accepted for security reasons
        (they appear in logs and browser history).

        Args:
            websocket: The WebSocket connection object

        Returns:
            The extracted token or None if not found
        """
        try:
            # Try newer websockets API first (websockets 10+)
            if hasattr(websocket, 'request') and hasattr(websocket.request, 'headers'):
                headers = websocket.request.headers
            elif hasattr(websocket, 'request_headers'):
                headers = websocket.request_headers
            else:
                return None

            # Only accept Authorization: Bearer header (not query params)
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header[7:]

            return None
        except Exception as e:
            logger.debug(f"Could not extract WebSocket token: {e}")
            return None

    def _validate_ws_auth(self, websocket, loop_id: str = "") -> bool:
        """Validate WebSocket authentication.

        Args:
            websocket: The WebSocket connection object
            loop_id: Optional loop_id for token validation

        Returns:
            True if authenticated or auth is disabled, False otherwise
        """
        if not auth_config.enabled:
            return True

        token = self._extract_ws_token(websocket)
        if not token:
            return False

        return auth_config.validate_token(token, loop_id)

    def _cleanup_stale_entries(self) -> None:
        """Remove stale entries from all tracking dicts."""
        now = time.time()
        cleaned_count = 0

        # Cleanup rate limiters
        self._cleanup_stale_rate_limiters()

        # Cleanup active_loops older than TTL
        with self._active_loops_lock:
            stale = [k for k, v in self._active_loops_last_access.items()
                     if now - v > self._ACTIVE_LOOPS_TTL]
            for k in stale:
                self.active_loops.pop(k, None)
                self._active_loops_last_access.pop(k, None)
                cleaned_count += 1

        # Cleanup debate_states older than TTL (only ended debates)
        with self._debate_states_lock:
            stale = [k for k, state in self.debate_states.items()
                     if state.get("ended") and
                        now - self._debate_states_last_access.get(k, 0) > self._DEBATE_STATES_TTL]
            for k in stale:
                self.debate_states.pop(k, None)
                self._debate_states_last_access.pop(k, None)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} stale entries")

    def _update_debate_state(self, event: StreamEvent) -> None:
        """Update cached debate state based on emitted events."""
        loop_id = event.loop_id
        with self._debate_states_lock:
            if event.type == StreamEventType.DEBATE_START:
                # Enforce max size with LRU eviction (only evict ended debates)
                if len(self.debate_states) >= self._MAX_DEBATE_STATES:
                    # Find oldest ended debate to evict
                    ended_states = [(k, self._debate_states_last_access.get(k, 0))
                                    for k, v in self.debate_states.items() if v.get("ended")]
                    if ended_states:
                        oldest = min(ended_states, key=lambda x: x[1])[0]
                        self.debate_states.pop(oldest, None)
                        self._debate_states_last_access.pop(oldest, None)
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
                self._debate_states_last_access[loop_id] = time.time()
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
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.CONSENSUS:
                if loop_id in self.debate_states:
                    state = self.debate_states[loop_id]
                    state["consensus_reached"] = event.data["reached"]
                    state["consensus_confidence"] = event.data["confidence"]
                    state["consensus_answer"] = event.data["answer"]
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.DEBATE_END:
                if loop_id in self.debate_states:
                    state = self.debate_states[loop_id]
                    state["ended"] = True
                    state["duration"] = event.data["duration"]
                    state["rounds"] = event.data["rounds"]
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.LOOP_UNREGISTER:
                self.debate_states.pop(loop_id, None)
                self._debate_states_last_access.pop(loop_id, None)

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
            except Exception as e:
                logger.debug(f"Client disconnected during broadcast: {e}")
                disconnected.add(client)

        self.clients -= disconnected

    async def broadcast_batch(self, events: list[StreamEvent]) -> None:
        """Send multiple events to all connected clients in a single message.

        Batching reduces WebSocket overhead by sending events as a JSON array
        instead of individual messages. Frontends should handle both single
        events and arrays for backward compatibility.

        Args:
            events: List of events to broadcast together
        """
        if not self.clients or not events:
            return

        # Send as JSON array for batching efficiency
        message = json.dumps([e.to_dict() for e in events])
        disconnected = set()

        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.debug(f"Client disconnected during batch broadcast: {e}")
                disconnected.add(client)

        self.clients -= disconnected

    async def _drain_loop(self) -> None:
        """Background task that drains the emitter queue and broadcasts.

        Uses batching to send multiple events in a single WebSocket message,
        reducing overhead and context switches by 5-10x for high-frequency
        event streams.
        """
        while self._running:
            # Collect all pending events into a batch
            events = list(self._emitter.drain())
            if events:
                # Send batch as a single message
                await self.broadcast_batch(events)
            await asyncio.sleep(0.05)

    def register_loop(self, loop_id: str, name: str, path: str = "") -> None:
        """Register a new nomic loop instance."""
        # Trigger periodic cleanup
        self._rate_limiter_cleanup_counter += 1
        if self._rate_limiter_cleanup_counter >= self._CLEANUP_INTERVAL:
            self._rate_limiter_cleanup_counter = 0
            self._cleanup_stale_entries()

        instance = LoopInstance(
            loop_id=loop_id,
            name=name,
            started_at=time.time(),
            path=path,
        )
        with self._active_loops_lock:
            # Enforce max size with LRU eviction
            if len(self.active_loops) >= self._MAX_ACTIVE_LOOPS:
                # Remove oldest by last access time
                oldest = min(self._active_loops_last_access, key=self._active_loops_last_access.get, default=None)
                if oldest:
                    self.active_loops.pop(oldest, None)
                    self._active_loops_last_access.pop(oldest, None)
            self.active_loops[loop_id] = instance
            self._active_loops_last_access[loop_id] = time.time()
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
                self._active_loops_last_access.pop(loop_id, None)
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

    def update_loop_state(self, loop_id: str, cycle: int | None = None, phase: str | None = None) -> None:
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

    def _extract_ws_origin(self, websocket) -> str:
        """Extract Origin header from websocket (handles different library versions)."""
        try:
            if hasattr(websocket, 'request') and hasattr(websocket.request, 'headers'):
                return websocket.request.headers.get("Origin", "")
            elif hasattr(websocket, 'request_headers'):
                return websocket.request_headers.get("Origin", "")
            return ""
        except Exception as e:
            logger.debug(f"Could not extract origin header: {e}")
            return ""

    def _validate_audience_payload(self, data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Validate audience message payload.

        Returns:
            Tuple of (validated_payload, error_message). If error, payload is None.
        """
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            return None, "Invalid payload format"

        # Limit payload size to 10KB (DoS protection)
        try:
            payload_str = json.dumps(payload)
            if len(payload_str) > 10240:
                return None, "Payload too large (max 10KB)"
        except (TypeError, ValueError):
            return None, "Invalid payload structure"

        return payload, None

    def _process_audience_message(
        self,
        msg_type: str,
        loop_id: str,
        payload: dict,
        client_id: str,
    ) -> None:
        """Process validated audience vote/suggestion message."""
        audience_msg = AudienceMessage(
            type="vote" if msg_type == "user_vote" else "suggestion",
            loop_id=loop_id,
            payload=payload,
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

        # Emit updated audience metrics after each vote
        if msg_type == "user_vote":
            metrics = self.audience_inbox.get_summary(loop_id=loop_id)
            self._emitter.emit(StreamEvent(
                type=StreamEventType.AUDIENCE_METRICS,
                data=metrics,
                loop_id=loop_id,
            ))

    async def handler(self, websocket) -> None:
        """Handle a WebSocket connection with origin validation."""
        # Validate origin for security
        origin = self._extract_ws_origin(websocket)
        if origin and origin not in WS_ALLOWED_ORIGINS:
            # Reject connection from unauthorized origin
            await websocket.close(4003, "Origin not allowed")
            return

        # Validate WebSocket authentication
        # Read operations are allowed without auth, but write operations require it
        is_authenticated = self._validate_ws_auth(websocket)

        # Generate cryptographically secure client ID (not predictable memory address)
        ws_id = id(websocket)
        client_id = secrets.token_urlsafe(16)
        # Enforce max size with LRU eviction
        if len(self._client_ids) >= self._MAX_CLIENT_IDS:
            self._client_ids.popitem(last=False)  # Remove oldest
        self._client_ids[ws_id] = client_id

        self.clients.add(websocket)
        logger.info(
            f"[ws] Client {client_id[:8]}... connected "
            f"(authenticated={is_authenticated}, total_clients={len(self.clients)})"
        )
        try:
            # Send connection info including auth status
            await websocket.send(json.dumps({
                "type": "connection_info",
                "data": {
                    "authenticated": is_authenticated,
                    "client_id": client_id[:8] + "...",  # Partial for privacy
                    "write_access": is_authenticated or not auth_config.enabled,
                }
            }))

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
                        loop = asyncio.get_running_loop()
                        data = await asyncio.wait_for(
                            loop.run_in_executor(None, json.loads, message),
                            timeout=5.0  # 5 second timeout for JSON parsing
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[ws] JSON parsing timed out - possible DoS attempt")
                        continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"[ws] Invalid JSON from client: {e}")
                        continue
                    except RuntimeError as e:
                        logger.error(f"[ws] Event loop error during JSON parsing: {e}")
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
                        # Require authentication for write operations
                        if not is_authenticated and auth_config.enabled:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Authentication required for voting/suggestions", "code": 401}
                            }))
                            continue

                        stored_client_id = self._client_ids.get(ws_id, secrets.token_urlsafe(16))
                        loop_id = data.get("loop_id", "")

                        # Validate loop_id exists and is active
                        if not loop_id or loop_id not in self.active_loops:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": f"Invalid or inactive loop_id: {loop_id}"}
                            }))
                            continue

                        # Validate payload
                        payload, error = self._validate_audience_payload(data)
                        if error:
                            await websocket.send(json.dumps({"type": "error", "data": {"message": error}}))
                            continue

                        # Rate limiting (thread-safe)
                        should_cleanup = False
                        with self._rate_limiters_lock:
                            if stored_client_id not in self._rate_limiters:
                                self._rate_limiters[stored_client_id] = TokenBucket(rate_per_minute=10.0, burst_size=5)
                            self._rate_limiter_last_access[stored_client_id] = time.time()
                            rate_limiter = self._rate_limiters[stored_client_id]
                            self._rate_limiter_cleanup_counter += 1
                            if self._rate_limiter_cleanup_counter >= self._CLEANUP_INTERVAL:
                                self._rate_limiter_cleanup_counter = 0
                                should_cleanup = True
                        if should_cleanup:
                            self._cleanup_stale_rate_limiters()

                        if not rate_limiter.consume(1):
                            await websocket.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Rate limited. Please wait before submitting again."}
                            }))
                            continue

                        # Process the message
                        self._process_audience_message(msg_type, loop_id, payload, stored_client_id)
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
                pass  # Normal disconnect, logged in finally block
            else:
                # Log unexpected errors for debugging (but don't expose to client)
                logger.error(f"[ws] Unexpected error for client {client_id[:8]}...: {error_name}: {e}")
        finally:
            self.clients.discard(websocket)
            logger.info(
                f"[ws] Client {client_id[:8]}... disconnected "
                f"(remaining_clients={len(self.clients)})"
            )
            # Clean up secure client ID mapping and rate limiters
            stored_client_id = self._client_ids.pop(ws_id, None)
            if stored_client_id:
                with self._rate_limiters_lock:
                    self._rate_limiters.pop(stored_client_id, None)
                    self._rate_limiter_last_access.pop(stored_client_id, None)

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        self._running = True
        # Store task reference and add error callback to prevent silent failures
        self._drain_task = asyncio.create_task(self._drain_loop())
        self._drain_task.add_done_callback(self._handle_drain_task_error)

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            max_size=WS_MAX_MESSAGE_SIZE,
            ping_interval=30,  # Send ping every 30s
            ping_timeout=10,   # Close connection if no pong within 10s
        ):
            logger.info(f"WebSocket server: ws://{self.host}:{self.port} (max message size: {WS_MAX_MESSAGE_SIZE} bytes)")
            await asyncio.Future()  # Run forever

    def stop(self) -> None:
        """Stop the server."""
        self._running = False

    def _handle_drain_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from the drain loop task."""
        try:
            exc = task.exception()
            if exc is not None:
                logger.error(f"Drain loop task failed with exception: {exc}")
        except asyncio.CancelledError:
            pass  # Task was cancelled, not an error

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

        # Shutdown debate executor
        global _debate_executor
        with _debate_executor_lock:
            if _debate_executor:
                logger.info("Shutting down debate executor...")
                _debate_executor.shutdown(wait=True, cancel_futures=False)
                _debate_executor = None


# create_arena_hooks is now imported from aragora.server.stream.arena_hooks


# =============================================================================
# Unified HTTP + WebSocket Server (aiohttp-based)
# =============================================================================

class AiohttpUnifiedServer(StreamAPIHandlersMixin):
    """
    Unified server using aiohttp to handle both HTTP API and WebSocket on a single port.

    This is the recommended server for production as it avoids CORS issues with
    separate ports for HTTP and WebSocket.

    HTTP API handlers are provided by StreamAPIHandlersMixin from stream_handlers.py.

    Lock Hierarchy (acquire in this order to prevent deadlocks):
    ---------------------------------------------------------------
    1. _rate_limiters_lock   - Protects _rate_limiters and _rate_limiter_last_access
    2. _debate_states_lock   - Protects debate_states and _debate_states_last_access
    3. _active_loops_lock    - Protects active_loops and _active_loops_last_access
    4. _cartographers_lock   - Protects cartographers registry

    IMPORTANT: Each method should only acquire ONE lock at a time. If multiple
    locks must be acquired (e.g., in cleanup methods), acquire them sequentially
    in the order above, releasing each before acquiring the next. Never nest
    lock acquisitions to prevent deadlocks.

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

        # Audience participation - Lock hierarchy level 1 (acquire first)
        self.audience_inbox = AudienceInbox()
        self._rate_limiters: dict[str, TokenBucket] = {}
        self._rate_limiter_last_access: dict[str, float] = {}
        self._rate_limiters_lock = threading.Lock()  # Lock #1 in hierarchy
        self._rate_limiter_cleanup_counter = 0
        self._RATE_LIMITER_TTL = 3600  # 1 hour TTL
        self._CLEANUP_INTERVAL = 100  # Cleanup every N accesses

        # Debate state caching - Lock hierarchy level 2
        self.debate_states: dict[str, dict] = {}
        self._debate_states_lock = threading.Lock()  # Lock #2 in hierarchy
        self._debate_states_last_access: dict[str, float] = {}
        self._DEBATE_STATES_TTL = 3600  # 1 hour TTL for ended debates
        self._MAX_DEBATE_STATES = 500  # Max cached states

        # Multi-loop tracking - Lock hierarchy level 3
        self.active_loops: dict[str, LoopInstance] = {}
        self._active_loops_lock = threading.Lock()  # Lock #3 in hierarchy
        self._active_loops_last_access: dict[str, float] = {}
        self._ACTIVE_LOOPS_TTL = 86400  # 24 hour TTL for stale loops
        self._MAX_ACTIVE_LOOPS = 1000  # Max concurrent loops

        # ArgumentCartographer registry - Lock hierarchy level 4 (acquire last)
        self.cartographers: Dict[str, Any] = {}
        self._cartographers_lock = threading.Lock()  # Lock #4 in hierarchy

        # Secure client ID mapping with LRU eviction
        self._client_ids: OrderedDict[int, str] = OrderedDict()
        self._MAX_CLIENT_IDS = 10000  # Max tracked clients

        # Optional stores (initialized from nomic_dir)
        self.elo_system = None
        self.insight_store = None
        self.flip_detector = None
        self.persona_manager = None
        self.debate_embeddings = None

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
            insights_path = nomic_dir / DB_INSIGHTS_PATH
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
            personas_path = nomic_dir / DB_PERSONAS_PATH
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

    def _cleanup_stale_entries(self) -> None:
        """Remove stale entries from all tracking dicts."""
        now = time.time()
        cleaned_count = 0

        # Cleanup rate limiters
        with self._rate_limiters_lock:
            stale = [k for k, v in self._rate_limiter_last_access.items()
                     if now - v > self._RATE_LIMITER_TTL]
            for k in stale:
                self._rate_limiters.pop(k, None)
                self._rate_limiter_last_access.pop(k, None)
                cleaned_count += 1

        # Cleanup active_loops older than TTL
        with self._active_loops_lock:
            stale = [k for k, v in self._active_loops_last_access.items()
                     if now - v > self._ACTIVE_LOOPS_TTL]
            for k in stale:
                self.active_loops.pop(k, None)
                self._active_loops_last_access.pop(k, None)
                cleaned_count += 1

        # Cleanup debate_states older than TTL (only ended debates)
        with self._debate_states_lock:
            stale = [k for k, state in self.debate_states.items()
                     if state.get("ended") and
                        now - self._debate_states_last_access.get(k, 0) > self._DEBATE_STATES_TTL]
            for k in stale:
                self.debate_states.pop(k, None)
                self._debate_states_last_access.pop(k, None)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} stale entries")

    def _update_debate_state(self, event: StreamEvent) -> None:
        """Update cached debate state based on emitted events."""
        loop_id = event.loop_id
        with self._debate_states_lock:
            if event.type == StreamEventType.DEBATE_START:
                # Enforce max size with LRU eviction (only evict ended debates)
                if len(self.debate_states) >= self._MAX_DEBATE_STATES:
                    ended_states = [(k, self._debate_states_last_access.get(k, 0))
                                    for k, v in self.debate_states.items() if v.get("ended")]
                    if ended_states:
                        oldest = min(ended_states, key=lambda x: x[1])[0]
                        self.debate_states.pop(oldest, None)
                        self._debate_states_last_access.pop(oldest, None)
                self.debate_states[loop_id] = {
                    "id": loop_id,
                    "task": event.data.get("task"),
                    "agents": event.data.get("agents"),
                    "started_at": event.timestamp,
                    "ended": False,
                }
                self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.DEBATE_END:
                if loop_id in self.debate_states:
                    self.debate_states[loop_id]["ended"] = True
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.LOOP_UNREGISTER:
                self.debate_states.pop(loop_id, None)
                self._debate_states_last_access.pop(loop_id, None)

    def register_loop(self, loop_id: str, name: str, path: str = "") -> None:
        """Register a new nomic loop instance."""
        # Trigger periodic cleanup
        self._rate_limiter_cleanup_counter += 1
        if self._rate_limiter_cleanup_counter >= self._CLEANUP_INTERVAL:
            self._rate_limiter_cleanup_counter = 0
            self._cleanup_stale_entries()

        with self._active_loops_lock:
            # Enforce max size with LRU eviction
            if len(self.active_loops) >= self._MAX_ACTIVE_LOOPS:
                oldest = min(self._active_loops_last_access, key=self._active_loops_last_access.get, default=None)
                if oldest:
                    self.active_loops.pop(oldest, None)
                    self._active_loops_last_access.pop(oldest, None)
            self.active_loops[loop_id] = LoopInstance(
                loop_id=loop_id,
                name=name,
                started_at=time.time(),
                path=path,
            )
            self._active_loops_last_access[loop_id] = time.time()
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
            self._active_loops_last_access.pop(loop_id, None)
        # Also cleanup associated cartographer to prevent memory leak
        self.unregister_cartographer(loop_id)
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
                    except Exception as e:
                        logger.debug(f"Skipping corrupted tournament file: {e}")
                        continue

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

            db_path = self.nomic_dir / DB_PERSONAS_PATH if self.nomic_dir else DB_PERSONAS_PATH
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
                        try:
                            meta = json.loads(meta_file.read_text())
                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse replay meta %s: %s", meta_file, e)
                            meta = {}
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
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning("Failed to parse event line: %s", e)
                            continue

            # Create artifact from events
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse replay meta %s: %s", meta_file, e)
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

    def _parse_debate_request(self, data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Parse and validate debate request data.

        Returns:
            Tuple of (parsed_config, error_message). If error_message is set,
            parsed_config will be None.
        """
        # Validate required fields with length limits
        question = data.get('question', '').strip()
        if not question:
            return None, "question field is required"
        if len(question) > 10000:
            return None, "question must be under 10,000 characters"

        # Parse optional fields with validation
        agents_str = data.get('agents', 'anthropic-api,openai-api,gemini,grok')
        try:
            rounds = min(max(int(data.get('rounds', 3)), 1), 10)  # Clamp to 1-10
        except (ValueError, TypeError):
            rounds = 3
        consensus = data.get('consensus', 'majority')

        return {
            "question": question,
            "agents_str": agents_str,
            "rounds": rounds,
            "consensus": consensus,
            "use_trending": data.get('use_trending', False),
            "trending_category": data.get('trending_category', None),
        }, None

    async def _fetch_trending_topic_async(self, category: Optional[str] = None) -> Optional[Any]:
        """Fetch a trending topic for the debate.

        Returns:
            A TrendingTopic object or None if unavailable.
        """
        try:
            from aragora.pulse.ingestor import (
                PulseManager,
                TwitterIngestor,
                HackerNewsIngestor,
                RedditIngestor,
            )

            manager = PulseManager()
            manager.add_ingestor("twitter", TwitterIngestor())
            manager.add_ingestor("hackernews", HackerNewsIngestor())
            manager.add_ingestor("reddit", RedditIngestor())

            filters = {}
            if category:
                filters["categories"] = [category]

            topics = await manager.get_trending_topics(
                limit_per_platform=3, filters=filters if filters else None
            )
            topic = manager.select_topic_for_debate(topics)

            if topic:
                logger.info(f"Selected trending topic: {topic.topic}")
            return topic
        except Exception as e:
            logger.warning(f"Trending topic fetch failed (non-fatal): {e}")
            return None

    def _execute_debate_thread(
        self,
        debate_id: str,
        question: str,
        agents_str: str,
        rounds: int,
        consensus: str,
        trending_topic: Optional[Any],
    ) -> None:
        """Execute a debate in a background thread.

        This method is run in a ThreadPoolExecutor to avoid blocking the event loop.
        """
        import asyncio as _asyncio

        try:
            # Parse agents with bounds check
            agent_list = [s.strip() for s in agents_str.split(",") if s.strip()]
            if len(agent_list) > MAX_AGENTS_PER_DEBATE:
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "error"
                    _active_debates[debate_id]["error"] = f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}"
                    _active_debates[debate_id]["completed_at"] = time.time()
                return
            if len(agent_list) < 2:
                with _active_debates_lock:
                    _active_debates[debate_id]["status"] = "error"
                    _active_debates[debate_id]["error"] = "At least 2 agents required for a debate"
                    _active_debates[debate_id]["completed_at"] = time.time()
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
                trending_topic=trending_topic,
            )

            # Run debate with timeout protection (10 minutes max)
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "running"

            async def run_with_timeout():
                return await _asyncio.wait_for(arena.run(), timeout=600)

            result = _asyncio.run(run_with_timeout())
            with _active_debates_lock:
                _active_debates[debate_id]["status"] = "completed"
                _active_debates[debate_id]["completed_at"] = time.time()
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
                _active_debates[debate_id]["completed_at"] = time.time()
                _active_debates[debate_id]["error"] = safe_msg
            logger.error(f"[debate] Thread error in {debate_id}: {str(e)}\n{error_trace}")
            # Emit error event to client
            self.emitter.emit(StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": safe_msg, "debate_id": debate_id},
            ))

    async def _handle_start_debate(self, request) -> 'aiohttp.web.Response':
        """POST /api/debate - Start an ad-hoc debate with specified question.

        Accepts JSON body with:
            question: The topic/question to debate (required)
            agents: Comma-separated agent list (optional, default: "anthropic-api,openai-api,gemini,grok")
            rounds: Number of debate rounds (optional, default: 3)
            consensus: Consensus method (optional, default: "majority")
            use_trending: If true, fetch a trending topic to seed the debate (optional)
            trending_category: Filter trending topics by category (optional)

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
        except Exception as e:
            logger.debug(f"Invalid JSON in request: {e}")
            return web.json_response(
                {"error": "Invalid JSON"},
                status=400,
                headers=self._cors_headers(origin)
            )

        # Parse and validate request
        config, error = self._parse_debate_request(data)
        if error:
            return web.json_response(
                {"error": error},
                status=400,
                headers=self._cors_headers(origin)
            )

        question = config["question"]
        agents_str = config["agents_str"]
        rounds = config["rounds"]
        consensus = config["consensus"]

        # Fetch trending topic if requested
        trending_topic = None
        if config["use_trending"]:
            trending_topic = await self._fetch_trending_topic_async(config["trending_category"])

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

        # Periodic cleanup of stale debates (every 100 debates)
        # Use lock to prevent race condition on counter
        should_cleanup = False
        with _debate_cleanup_counter_lock:
            global _debate_cleanup_counter
            _debate_cleanup_counter += 1
            if _debate_cleanup_counter >= 100:
                _debate_cleanup_counter = 0
                should_cleanup = True
        if should_cleanup:
            _cleanup_stale_debates_stream()

        # Set loop_id on emitter so events are tagged
        self.emitter.set_loop_id(debate_id)

        # Use thread pool to prevent unbounded thread creation
        _debate_executor = get_debate_executor()
        with _debate_executor_lock:
            if _debate_executor is None:
                _debate_executor = ThreadPoolExecutor(
                    max_workers=MAX_CONCURRENT_DEBATES,
                    thread_name_prefix="debate-"
                )
                set_debate_executor(_debate_executor)
            executor = _debate_executor

        try:
            executor.submit(
                self._execute_debate_thread,
                debate_id, question, agents_str, rounds, consensus, trending_topic
            )
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

    def _validate_audience_payload(self, data: dict) -> tuple[Optional[dict], Optional[str]]:
        """Validate audience message payload.

        Returns:
            Tuple of (validated_payload, error_message). If error, payload is None.
        """
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            return None, "Invalid payload format"

        # Limit payload size to 10KB (DoS protection)
        try:
            payload_str = json.dumps(payload)
            if len(payload_str) > 10240:
                return None, "Payload too large (max 10KB)"
        except (TypeError, ValueError):
            return None, "Invalid payload structure"

        return payload, None

    def _process_audience_message(
        self,
        msg_type: str,
        loop_id: str,
        payload: dict,
        client_id: str,
    ) -> None:
        """Process validated audience vote/suggestion message."""
        audience_msg = AudienceMessage(
            type="vote" if msg_type == "user_vote" else "suggestion",
            loop_id=loop_id,
            payload=payload,
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

        # Emit updated audience metrics after each vote
        if msg_type == "user_vote":
            metrics = self.audience_inbox.get_summary(loop_id=loop_id)
            self._emitter.emit(StreamEvent(
                type=StreamEventType.AUDIENCE_METRICS,
                data=metrics,
                loop_id=loop_id,
            ))

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
                client_ip = remote_ip  # Default to direct connection IP
                if remote_ip in TRUSTED_PROXIES:
                    # Only trust X-Forwarded-For from trusted proxies
                    forwarded = request.headers.get('X-Forwarded-For', '')
                    if forwarded:
                        first_ip = forwarded.split(',')[0].strip()
                        if first_ip:
                            client_ip = first_ip

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

        # Initialize tracking variables before any operations that could fail
        ws_id = id(ws)
        client_id = secrets.token_hex(16)
        self.clients.add(ws)
        # Enforce max size with LRU eviction
        if len(self._client_ids) >= self._MAX_CLIENT_IDS:
            self._client_ids.popitem(last=False)  # Remove oldest
        self._client_ids[ws_id] = client_id

        # Initialize rate limiter for this client (thread-safe)
        with self._rate_limiters_lock:
            self._rate_limiters[client_id] = TokenBucket(
                rate_per_minute=10.0,  # 10 messages per minute
                burst_size=5  # Allow burst of 5
            )
            self._rate_limiter_last_access[client_id] = time.time()

        logger.info(
            f"[ws] Client {client_id[:8]}... connected "
            f"(total_clients={len(self.clients)})"
        )

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
                            # Proprioceptive Socket: Use ws-bound loop_id as fallback
                            loop_id = data.get("loop_id") or getattr(ws, '_bound_loop_id', "")

                            # Optional per-message token validation
                            msg_token = data.get("token")
                            if msg_token:
                                from aragora.server.auth import auth_config
                                if not auth_config.validate_token(msg_token, loop_id):
                                    await ws.send_json({"type": "error", "data": {"code": "AUTH_FAILED", "message": "Invalid or revoked token"}})
                                    continue

                            # Validate loop_id exists and is active
                            with self._active_loops_lock:
                                loop_valid = loop_id and loop_id in self.active_loops
                            if not loop_valid:
                                await ws.send_json({"type": "error", "data": {"message": f"Invalid or inactive loop_id: {loop_id}"}})
                                continue

                            # Proprioceptive Socket: Bind loop_id to WebSocket for future reference
                            ws._bound_loop_id = loop_id

                            # Validate payload
                            payload, error = self._validate_audience_payload(data)
                            if error:
                                await ws.send_json({"type": "error", "data": {"message": error}})
                                continue

                            # Check rate limit (thread-safe)
                            with self._rate_limiters_lock:
                                self._rate_limiter_last_access[client_id] = time.time()
                                rate_limiter = self._rate_limiters.get(client_id)
                            if rate_limiter is None or not rate_limiter.consume(1):
                                await ws.send_json({"type": "error", "data": {"message": "Rate limit exceeded, try again later"}})
                                continue

                            # Process the message
                            self._process_audience_message(msg_type, loop_id, payload, client_id)
                            await ws.send_json({"type": "ack", "data": {"message": "Message received", "msg_type": msg_type}})

                    except json.JSONDecodeError as e:
                        logger.warning(f"[ws] Invalid JSON: {e.msg} at pos {e.pos}")
                        await ws.send_json({
                            "type": "error",
                            "data": {"code": "INVALID_JSON", "message": f"JSON parse error: {e.msg}"}
                        })

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'[ws] Error: {ws.exception()}')
                    break

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    logger.debug(f'[ws] Client {client_id[:8]}... closed connection')
                    break

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    logger.warning(f'[ws] Binary message rejected from {client_id[:8]}...')
                    await ws.send_json({
                        "type": "error",
                        "data": {"code": "BINARY_NOT_SUPPORTED", "message": "Binary messages not supported"}
                    })

                # PING/PONG handled automatically by aiohttp, but log if we see them
                elif msg.type in (aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG):
                    pass  # Handled by aiohttp automatically

                else:
                    logger.warning(f'[ws] Unhandled message type: {msg.type}')

        finally:
            self.clients.discard(ws)
            self._client_ids.pop(ws_id, None)
            # Clean up rate limiter for this client (thread-safe)
            with self._rate_limiters_lock:
                self._rate_limiters.pop(client_id, None)
                self._rate_limiter_last_access.pop(client_id, None)
            logger.info(
                f"[ws] Client {client_id[:8]}... disconnected "
                f"(remaining_clients={len(self.clients)})"
            )

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
                    except Exception as e:
                        logger.debug("WebSocket client disconnected during broadcast: %s", type(e).__name__)
                        dead_clients.append(client)

                if dead_clients:
                    logger.info("Removed %d dead WebSocket client(s)", len(dead_clients))
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

        logger.info(f"Unified server (HTTP+WS) running on http://{self.host}:{self.port}")
        logger.info(f"  WebSocket: ws://{self.host}:{self.port}/")
        logger.info(f"  HTTP API:  http://{self.host}:{self.port}/api/*")

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
