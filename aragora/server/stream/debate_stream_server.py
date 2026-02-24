"""
WebSocket-only debate streaming server using the websockets library.

Provides real-time debate event streaming to connected clients without HTTP API.
For combined HTTP+WebSocket, use AiohttpUnifiedServer instead.

Usage:
    server = DebateStreamServer(port=8765)
    hooks = create_arena_hooks(server.emitter)
    arena = Arena(env, agents, event_hooks=hooks)

    # In async context:
    asyncio.create_task(server.start())
    await arena.run()
"""

import asyncio
import json
import logging
import os
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from .emitter import TokenBucket
from .events import AudienceMessage, StreamEvent, StreamEventType
from .replay_buffer import ConnectionQualityTracker, EventReplayBuffer
from .server_base import ServerBase
from .state_manager import LoopInstance

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _timeout_context(timeout_seconds: float):
    """Compatibility wrapper for asyncio.timeout (Python 3.11+) using asyncio.wait_for.

    Provides a context manager that can be used with async with for timeout handling.
    On Python 3.11+, uses asyncio.timeout(). On earlier versions, uses asyncio.wait_for().
    """
    if sys.version_info >= (3, 11):
        # Python 3.11+ has asyncio.timeout
        async with asyncio.timeout(timeout_seconds):
            yield
    else:
        # Python 3.10 fallback: create a task-based timeout context
        try:
            yield
        except asyncio.TimeoutError:
            raise


# Import centralized config
from aragora.config import WS_MAX_MESSAGE_SIZE

# Import auth for WebSocket authentication
from aragora.server.auth import auth_config

# Centralized CORS configuration
from aragora.server.cors_config import WS_ALLOWED_ORIGINS

# Trusted proxies for X-Forwarded-For header validation
TRUSTED_PROXIES = frozenset(
    p.strip() for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
)

# Connection rate limiting per IP
WS_CONNECTIONS_PER_IP_PER_MINUTE = int(os.getenv("ARAGORA_WS_CONN_RATE", "30"))

# Global rate limit disable for testing/load tests
WS_RATE_LIMITING_DISABLED = os.environ.get("ARAGORA_DISABLE_ALL_RATE_LIMITS", "").lower() in (
    "1",
    "true",
    "yes",
)

# Token revalidation interval for long-lived connections (5 minutes)
WS_TOKEN_REVALIDATION_INTERVAL = 300.0

# Maximum connections per IP (concurrent)
WS_MAX_CONNECTIONS_PER_IP = int(os.getenv("ARAGORA_WS_MAX_PER_IP", "10"))

# Per-connection message rate limiting (messages per second)
WS_MESSAGES_PER_SECOND = int(os.getenv("ARAGORA_WS_MSG_RATE", "10"))
WS_MESSAGE_BURST_SIZE = int(os.getenv("ARAGORA_WS_MSG_BURST", "20"))


class WebSocketMessageRateLimiter:
    """Per-connection message rate limiter using token bucket algorithm.

    Limits the rate of incoming messages from a single WebSocket connection
    to prevent DoS attacks via message flooding.
    """

    def __init__(
        self,
        messages_per_second: float = WS_MESSAGES_PER_SECOND,
        burst_size: int = WS_MESSAGE_BURST_SIZE,
    ):
        """Initialize rate limiter.

        Args:
            messages_per_second: Maximum sustained message rate
            burst_size: Maximum burst of messages allowed
        """
        self.messages_per_second = messages_per_second
        self.burst_size = burst_size
        self._tokens = float(burst_size)
        self._last_update = time.time()

    def allow_message(self) -> bool:
        """Check if a message is allowed under the rate limit.

        Returns:
            True if message is allowed, False if rate limited
        """
        # Bypass rate limiting if globally disabled (for load tests)
        if WS_RATE_LIMITING_DISABLED:
            return True

        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # Refill tokens based on elapsed time
        self._tokens = min(self.burst_size, self._tokens + elapsed * self.messages_per_second)

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class DebateStreamServer(ServerBase):
    """
    WebSocket server broadcasting debate events to connected clients.

    Supports multiple concurrent nomic loop instances with view switching.
    Inherits common functionality from ServerBase including rate limiting,
    debate state caching, and active loops tracking.

    This server uses the pure websockets library for WebSocket handling.
    For combined HTTP+WebSocket support, use AiohttpUnifiedServer instead.

    Usage:
        server = DebateStreamServer(port=8765)
        hooks = create_arena_hooks(server.emitter)
        arena = Arena(env, agents, event_hooks=hooks)

        # In async context:
        asyncio.create_task(server.start())
        await arena.run()
    """

    # Cleanup interval for rate limiters
    _CLEANUP_INTERVAL = 100

    # Application-level heartbeat interval (seconds).
    # WebSocket-level ping/pong handles connection liveness, but frontends
    # cannot observe those frames.  This sends a visible JSON heartbeat so
    # the client can detect stalls without guessing.
    _HEARTBEAT_INTERVAL_S = 15.0

    def __init__(self, host: str = "localhost", port: int = 8765, enable_tts: bool = False):
        # Initialize base class with common functionality
        super().__init__()

        self.host = host
        self.port = port
        self.current_debate: dict | None = None
        self.enable_tts = enable_tts

        # TTS event bridge (lazily created when enable_tts is True)
        self._tts_bridge: Any | None = None

        # WebSocket-specific: connection rate limiting per IP
        self._ws_conn_rate: dict[str, list[float]] = {}  # ip -> list of connection timestamps
        self._ws_conn_rate_lock = threading.Lock()
        self._ws_conn_per_ip: dict[str, int] = {}  # ip -> current connection count

        # Token revalidation tracking for long-lived connections
        self._ws_token_validated: dict[int, float] = {}  # ws_id -> last validation time

        # Per-connection message rate limiters
        self._ws_msg_limiters: dict[int, WebSocketMessageRateLimiter] = {}  # ws_id -> rate limiter

        # Async lock for client set operations (prevents race conditions during broadcast)
        self._clients_lock = asyncio.Lock()

        # Track which debate each client is subscribed to (for stream isolation)
        # Key: ws_id (int), Value: debate_id (str)
        # SECURITY: Prevents data leakage between concurrent debates
        self._client_subscriptions: dict[int, str] = {}

        # Event replay buffer for reconnection support
        self._replay_buffer = EventReplayBuffer()
        set_global_replay_buffer(self._replay_buffer)

        # Connection quality tracker for per-client metrics
        self._quality_tracker = ConnectionQualityTracker()

        # Stop event for graceful shutdown
        self._stop_event: asyncio.Event | None = None

    def _cleanup_stale_rate_limiters(self) -> None:
        """Remove rate limiters not accessed within TTL period."""
        self.cleanup_rate_limiters()

    def _extract_ws_token(self, websocket) -> str | None:
        """Extract authentication token from WebSocket connection.

        Attempts to extract token from:
        1. Authorization: Bearer header (preferred for server-to-server)
        2. Sec-WebSocket-Protocol header with 'access_token.' prefix (for browsers)

        Query parameter tokens are not accepted for security reasons
        (they appear in logs and browser history).

        Args:
            websocket: The WebSocket connection object

        Returns:
            The extracted token or None if not found
        """
        try:
            # Try newer websockets API first (websockets 10+)
            if hasattr(websocket, "request") and hasattr(websocket.request, "headers"):
                headers = websocket.request.headers
            elif hasattr(websocket, "request_headers"):
                headers = websocket.request_headers
            else:
                return None

            # Method 1: Authorization: Bearer header (server-to-server, CLI)
            auth_header = headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                return auth_header[7:]

            # Method 2: Sec-WebSocket-Protocol with token (browser clients)
            # Browsers can't set custom headers, but can set subprotocol
            # Format: "access_token.{jwt}" or "aragora-v1, access_token.{jwt}"
            protocol_header = headers.get("Sec-WebSocket-Protocol", "")
            if protocol_header:
                for protocol in protocol_header.split(","):
                    protocol = protocol.strip()
                    if protocol.startswith("access_token."):
                        return protocol[13:]  # len("access_token.") = 13

            return None
        except (AttributeError, KeyError, TypeError) as e:
            logger.debug("Could not extract WebSocket token: %s", e)
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

    def _extract_ws_ip(self, websocket) -> str:
        """Extract client IP address from WebSocket connection.

        Handles X-Forwarded-For header from trusted proxies.

        Args:
            websocket: The WebSocket connection object

        Returns:
            Client IP address string
        """
        try:
            # Get direct connection IP
            if hasattr(websocket, "remote_address"):
                direct_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
            else:
                direct_ip = "unknown"

            # Check X-Forwarded-For if from trusted proxy
            if direct_ip in TRUSTED_PROXIES:
                headers = None
                if hasattr(websocket, "request") and hasattr(websocket.request, "headers"):
                    headers = websocket.request.headers
                elif hasattr(websocket, "request_headers"):
                    headers = websocket.request_headers

                if headers:
                    xff = headers.get("X-Forwarded-For", "")
                    if xff:
                        # Take first IP (original client)
                        return xff.split(",")[0].strip()

            return direct_ip
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            logger.debug("Could not extract client IP: %s", e)
            return "unknown"

    def _check_ws_connection_rate(self, ip: str) -> tuple[bool, str]:
        """Check if IP is within WebSocket connection rate limit.

        Args:
            ip: Client IP address

        Returns:
            Tuple of (allowed: bool, error_message: str)
        """
        # Bypass rate limiting if globally disabled (for load tests)
        if WS_RATE_LIMITING_DISABLED:
            return True, ""

        if ip == "unknown":
            return True, ""  # Can't rate limit unknown IPs

        now = time.time()
        window_start = now - 60.0  # 1 minute window

        with self._ws_conn_rate_lock:
            # Clean old timestamps
            if ip in self._ws_conn_rate:
                self._ws_conn_rate[ip] = [ts for ts in self._ws_conn_rate[ip] if ts > window_start]
            else:
                self._ws_conn_rate[ip] = []

            # Check rate
            if len(self._ws_conn_rate[ip]) >= WS_CONNECTIONS_PER_IP_PER_MINUTE:
                return (
                    False,
                    f"Connection rate limit exceeded ({WS_CONNECTIONS_PER_IP_PER_MINUTE}/min)",
                )

            # Check concurrent connections
            current_count = self._ws_conn_per_ip.get(ip, 0)
            if current_count >= WS_MAX_CONNECTIONS_PER_IP:
                return False, f"Max concurrent connections exceeded ({WS_MAX_CONNECTIONS_PER_IP})"

            # Record connection
            self._ws_conn_rate[ip].append(now)
            self._ws_conn_per_ip[ip] = current_count + 1

            return True, ""

    def _release_ws_connection(self, ip: str) -> None:
        """Release a WebSocket connection slot for an IP.

        Args:
            ip: Client IP address
        """
        if ip == "unknown":
            return

        with self._ws_conn_rate_lock:
            current = self._ws_conn_per_ip.get(ip, 0)
            if current > 0:
                self._ws_conn_per_ip[ip] = current - 1

    def _should_revalidate_token(self, ws_id: int) -> bool:
        """Check if token should be revalidated for a connection.

        Args:
            ws_id: WebSocket connection ID

        Returns:
            True if token needs revalidation
        """
        last_validated = self._ws_token_validated.get(ws_id, 0)
        return (time.time() - last_validated) > WS_TOKEN_REVALIDATION_INTERVAL

    def _mark_token_validated(self, ws_id: int) -> None:
        """Mark token as validated for a connection.

        Args:
            ws_id: WebSocket connection ID
        """
        self._ws_token_validated[ws_id] = time.time()

    def _cleanup_stale_entries(self) -> None:
        """Remove stale entries from all tracking dicts.

        Delegates to ServerBase.cleanup_all() and adds server-specific cleanup.
        """
        results = self.cleanup_all()
        total = sum(results.values())
        if total > 0:
            logger.debug("Cleaned up %s stale entries", total)

    def _update_debate_state(self, event: StreamEvent) -> None:
        """Update cached debate state based on emitted events.

        Overrides ServerBase._update_debate_state with StreamEvent-specific handling.
        Also feeds every debate-scoped event into the replay buffer.
        """
        # Feed into replay buffer for reconnect support
        self._replay_buffer.append(event)

        loop_id = event.loop_id
        with self._debate_states_lock:
            if event.type == StreamEventType.DEBATE_START:
                # Enforce max size with LRU eviction (only evict ended debates)
                if len(self.debate_states) >= self.config.max_debate_states:
                    # Find oldest ended debate to evict
                    ended_states = [
                        (k, self._debate_states_last_access.get(k, 0))
                        for k, v in self.debate_states.items()
                        if v.get("ended")
                    ]
                    if ended_states:
                        oldest = min(ended_states, key=lambda x: x[1])[0]
                        self.debate_states.pop(oldest, None)
                        self._debate_states_last_access.pop(oldest, None)
                self.debate_states[loop_id] = {
                    "id": loop_id,
                    "task": event.data.get("task", ""),
                    "agents": event.data.get("agents", []),
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
                    msg_entry: dict = {
                        "agent": event.agent,
                        "role": event.data.get("role", "agent"),
                        "round": event.round,
                        "content": event.data.get("content", ""),
                    }
                    if event.data.get("confidence_score") is not None:
                        msg_entry["confidence_score"] = event.data["confidence_score"]
                    if event.data.get("reasoning_phase"):
                        msg_entry["reasoning_phase"] = event.data["reasoning_phase"]
                    state["messages"].append(msg_entry)
                    # Cap at last 1000 messages to allow full debate history without truncation
                    if len(state["messages"]) > 1000:
                        state["messages"] = state["messages"][-1000:]
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.CONSENSUS:
                if loop_id in self.debate_states:
                    state = self.debate_states[loop_id]
                    state["consensus_reached"] = event.data.get("reached", False)
                    state["consensus_confidence"] = event.data.get("confidence", 0.0)
                    state["consensus_answer"] = event.data.get("answer", "")
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.DEBATE_END:
                if loop_id in self.debate_states:
                    state = self.debate_states[loop_id]
                    state["ended"] = True
                    state["duration"] = event.data.get("duration", 0.0)
                    state["rounds"] = event.data.get("rounds", 0)
                    self._debate_states_last_access[loop_id] = time.time()
            elif event.type == StreamEventType.LOOP_UNREGISTER:
                self.debate_states.pop(loop_id, None)
                self._debate_states_last_access.pop(loop_id, None)
                self._replay_buffer.remove(loop_id)

        # Update loop state for cycle/phase events (outside debate_states_lock)
        if event.type == StreamEventType.CYCLE_START:
            self.update_loop_state(loop_id, cycle=event.data.get("cycle"))
        elif event.type == StreamEventType.PHASE_START:
            self.update_loop_state(loop_id, phase=event.data.get("phase"))

    async def broadcast(self, event: StreamEvent) -> None:
        """Send event only to clients subscribed to the event's debate.

        SECURITY: Filters events by subscription to prevent data leakage
        between concurrent debates.
        """
        if not self.clients:
            return

        message = event.to_json()
        event_loop_id = event.loop_id
        disconnected = set()

        # Take snapshot under lock to prevent race conditions
        async with self._clients_lock:
            clients_snapshot = list(self.clients)

        for client in clients_snapshot:
            client_ws_id = id(client)
            subscribed_id = self._client_subscriptions.get(client_ws_id)

            # Send if:
            # 1. Event has no loop_id (system-wide events like heartbeat)
            # 2. Client is subscribed to this specific debate
            # SECURITY: Unsubscribed clients (subscribed_id is None) do NOT
            # receive debate-scoped events to prevent data leakage.
            should_send = (
                not event_loop_id  # System events go to all
                or subscribed_id == event_loop_id  # Subscribed to this debate
            )

            if should_send:
                try:
                    # Timeout prevents hanging if client disconnects mid-send
                    try:
                        await asyncio.wait_for(client.send(message), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Client send timed out during broadcast, marking for disconnect"
                        )
                        disconnected.add(client)
                except (OSError, ConnectionError, RuntimeError, Exception) as e:
                    # WebSocket/network errors during send
                    logger.debug("Client disconnected during broadcast: %s", e)
                    disconnected.add(client)

        if disconnected:
            async with self._clients_lock:
                self.clients -= disconnected
            # Cleanup subscription tracking for disconnected clients
            for client in disconnected:
                self._client_subscriptions.pop(id(client), None)

    async def broadcast_batch(self, events: list[StreamEvent]) -> None:
        """Send multiple events only to subscribed clients.

        SECURITY: Filters events by subscription to prevent data leakage
        between concurrent debates.

        Batching reduces WebSocket overhead by sending events as a JSON array
        instead of individual messages. Frontends should handle both single
        events and arrays for backward compatibility.

        Args:
            events: List of events to broadcast together
        """
        if not self.clients or not events:
            return

        # Determine loop_id from first event (batches are typically same-debate)
        batch_loop_id = events[0].loop_id if events else None

        # Send as JSON array for batching efficiency
        message = json.dumps([e.to_dict() for e in events])
        disconnected = set()

        # Take snapshot under lock to prevent race conditions
        async with self._clients_lock:
            clients_snapshot = list(self.clients)

        for client in clients_snapshot:
            client_ws_id = id(client)
            subscribed_id = self._client_subscriptions.get(client_ws_id)

            # Send if subscribed to this debate
            # SECURITY: Unsubscribed clients do NOT receive debate-scoped events
            should_send = (
                not batch_loop_id  # System events go to all
                or subscribed_id == batch_loop_id  # Subscribed to this debate
            )

            if should_send:
                try:
                    # Timeout prevents hanging if client disconnects mid-send
                    try:
                        await asyncio.wait_for(client.send(message), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Client send timed out during batch broadcast, marking for disconnect"
                        )
                        disconnected.add(client)
                except (OSError, ConnectionError, RuntimeError, Exception) as e:
                    # WebSocket/network errors during send
                    logger.debug("Client disconnected during batch broadcast: %s", e)
                    disconnected.add(client)

        if disconnected:
            async with self._clients_lock:
                self.clients -= disconnected

    def _group_events_by_agent(self, events: list[StreamEvent]) -> list[StreamEvent]:
        """Reorder events to group TOKEN_DELTA by (agent, task_id) for smooth frontend rendering.

        Non-token events maintain their relative order and trigger token flushes.
        Token events are grouped by (agent, task_id), reducing visual interleaving during
        parallel agent generation. The task_id ensures concurrent outputs from the same
        agent (e.g., multiple critiques) are kept separate.

        NOTE: We intentionally do NOT sort by global seq after grouping. The frontend
        uses agent_seq for per-agent token ordering, which correctly handles
        out-of-order tokens. Sorting by global seq would undo the agent grouping
        and reintroduce interleaving.
        """
        result: list[StreamEvent] = []
        # Key by (agent, task_id) to distinguish concurrent outputs from same agent
        token_groups: dict[tuple[str, str], list[StreamEvent]] = {}

        for event in events:
            if event.type == StreamEventType.TOKEN_DELTA and event.agent:
                # Buffer token events by (agent, task_id)
                key = (event.agent, event.task_id or "")
                if key not in token_groups:
                    token_groups[key] = []
                token_groups[key].append(event)
            else:
                # Flush buffered tokens before non-token event
                # Sort each group's tokens by agent_seq for correct ordering
                for group_tokens in token_groups.values():
                    group_tokens.sort(key=lambda e: e.agent_seq if e.agent_seq else 0)
                    result.extend(group_tokens)
                token_groups.clear()
                result.append(event)

        # Flush remaining tokens (grouped by agent+task_id)
        # Sort each group's tokens by agent_seq for correct ordering
        for group_tokens in token_groups.values():
            group_tokens.sort(key=lambda e: e.agent_seq if e.agent_seq else 0)
            result.extend(group_tokens)

        return result

    async def _drain_loop(self) -> None:
        """Background task that drains the emitter queue and broadcasts.

        Token events are sent in small batches (3 tokens) for progressive display.
        Other events are batched normally for efficiency. This prevents large blocks
        of text appearing all at once when API responses are buffered.

        Broadcasts have timeouts to prevent slow/hung WebSocket clients from
        blocking the entire drain loop (which would cause event queue backup).
        """
        from aragora.config import STREAM_BATCH_SIZE, STREAM_DRAIN_INTERVAL_MS

        while self._running:
            events = list(self._emitter.drain(max_batch_size=STREAM_BATCH_SIZE))
            if events:
                # Separate token events for immediate delivery
                token_events = [e for e in events if e.type == StreamEventType.TOKEN_DELTA]
                other_events = [e for e in events if e.type != StreamEventType.TOKEN_DELTA]

                # Add timeout protection to prevent slow clients from blocking drain
                try:
                    # Send tokens progressively in small batches of 3 for efficiency
                    if token_events:
                        for i in range(0, len(token_events), 3):
                            batch = token_events[i : i + 3]
                            await asyncio.wait_for(
                                self.broadcast_batch(batch),
                                timeout=2.0,  # 2 second timeout per token batch
                            )

                    # Batch other events normally with agent grouping
                    if other_events:
                        grouped = self._group_events_by_agent(other_events)
                        await asyncio.wait_for(
                            self.broadcast_batch(grouped),
                            timeout=5.0,  # 5 second timeout for other events
                        )
                except asyncio.TimeoutError:
                    logger.warning(
                        "[ws] Broadcast timed out - slow client may be blocking. "
                        "Some events may be dropped to prevent queue backup."
                    )

            await asyncio.sleep(STREAM_DRAIN_INTERVAL_MS / 1000.0)

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat messages to all connected clients.

        Heartbeats are application-level JSON frames that let the frontend
        distinguish "server is alive but quiet" from "connection stalled".
        Each heartbeat includes ``last_seq`` for the client's subscribed
        debate so the frontend can detect missed events and request replay.
        """
        while self._running:
            await asyncio.sleep(self._HEARTBEAT_INTERVAL_S)
            if not self._running:
                break
            async with self._clients_lock:
                clients = list(self.clients)
            if not clients:
                continue

            base_data = {
                "server_time": time.time(),
                "active_debates": len(self.debate_states),
                "connected_clients": len(clients),
            }

            for ws in clients:
                try:
                    ws_id = id(ws)
                    data = dict(base_data)
                    # Include last_seq for the client's subscribed debate
                    sub_id = self._client_subscriptions.get(ws_id)
                    if sub_id:
                        data["last_seq"] = self._replay_buffer.get_latest_seq(sub_id)
                        data["buffer_size"] = self._replay_buffer.get_buffered_count(sub_id)
                        data["oldest_seq"] = self._replay_buffer.get_oldest_seq(sub_id)
                    # Include per-client connection quality summary
                    quality = self._quality_tracker.get_quality(ws_id)
                    if quality:
                        data["connection_quality"] = {
                            "reconnect_count": quality["reconnect_count"],
                            "avg_latency_ms": quality["avg_latency_ms"],
                            "uptime_seconds": quality["uptime_seconds"],
                        }
                    heartbeat = json.dumps({"type": "heartbeat", "data": data})
                    await asyncio.wait_for(ws.send(heartbeat), timeout=2.0)
                except (asyncio.TimeoutError, OSError, RuntimeError):
                    pass  # Skip slow/closed clients

    def register_loop(self, loop_id: str, name: str, path: str = "") -> None:
        """Register a new nomic loop instance."""
        # Trigger periodic cleanup using base class config
        self._rate_limiter_cleanup_counter += 1
        if self._rate_limiter_cleanup_counter >= self.config.rate_limiter_cleanup_interval:
            self._rate_limiter_cleanup_counter = 0
            self._cleanup_stale_entries()

        instance = LoopInstance(
            loop_id=loop_id,
            name=name,
            started_at=time.time(),
            path=path,
        )
        # Use base class method for active loop management
        self.set_active_loop(loop_id, instance)
        with self._active_loops_lock:
            loop_count = len(self.active_loops)
        # Emit registration event
        self._emitter.emit(
            StreamEvent(
                type=StreamEventType.LOOP_REGISTER,
                data={
                    "loop_id": loop_id,
                    "name": name,
                    "started_at": instance.started_at,
                    "path": path,
                    "active_loops": loop_count,
                },
            )
        )

    def unregister_loop(self, loop_id: str) -> None:
        """Unregister a nomic loop instance."""
        removed = self.remove_active_loop(loop_id)
        if removed is None:
            return  # Loop not found, nothing to unregister
        with self._active_loops_lock:
            loop_count = len(self.active_loops)
        # Emit unregistration event
        self._emitter.emit(
            StreamEvent(
                type=StreamEventType.LOOP_UNREGISTER,
                data={
                    "loop_id": loop_id,
                    "active_loops": loop_count,
                },
            )
        )

    def update_loop_state(
        self, loop_id: str, cycle: int | None = None, phase: str | None = None
    ) -> None:
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
            if hasattr(websocket, "request") and hasattr(websocket.request, "headers"):
                return websocket.request.headers.get("Origin", "")
            elif hasattr(websocket, "request_headers"):
                return websocket.request_headers.get("Origin", "")
            return ""
        except (AttributeError, KeyError, TypeError) as e:
            logger.debug("Could not extract origin header: %s", e)
            return ""

    def _validate_audience_payload(self, data: dict) -> tuple[dict | None, str | None]:
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
        event_type = (
            StreamEventType.USER_VOTE
            if msg_type == "user_vote"
            else StreamEventType.USER_SUGGESTION
        )
        self._emitter.emit(
            StreamEvent(
                type=event_type,
                data=audience_msg.payload,
                loop_id=loop_id,
            )
        )

        # Emit updated audience metrics after each vote
        if msg_type == "user_vote":
            metrics = self.audience_inbox.get_summary(loop_id=loop_id)
            self._emitter.emit(
                StreamEvent(
                    type=StreamEventType.AUDIENCE_METRICS,
                    data=metrics,
                    loop_id=loop_id,
                )
            )

    async def _setup_connection(self, websocket) -> tuple[bool, str, str, int, bool, str | None]:
        """Set up WebSocket connection with validation.

        Returns:
            Tuple of (success, client_ip, client_id, ws_id, is_authenticated, ws_token)
            If success is False, connection has been closed with error.
        """
        client_ip = self._extract_ws_ip(websocket)

        # Check connection rate limit
        rate_allowed, rate_error = self._check_ws_connection_rate(client_ip)
        if not rate_allowed:
            logger.warning("[ws] Connection rejected for %s: %s", client_ip, rate_error)
            await websocket.close(4029, rate_error)
            return (False, client_ip, "", 0, False, None)

        # Validate origin
        origin = self._extract_ws_origin(websocket)
        if origin and origin not in WS_ALLOWED_ORIGINS:
            self._release_ws_connection(client_ip)
            logger.warning("[ws] Origin not allowed for %s: %s", client_ip, origin)
            await websocket.close(4003, "Origin not allowed")
            return (False, client_ip, "", 0, False, None)

        # Validate authentication
        is_authenticated = self._validate_ws_auth(websocket)
        ws_token = self._extract_ws_token(websocket)

        # Generate secure client ID
        ws_id = id(websocket)
        client_id = secrets.token_urlsafe(16)
        if len(self._client_ids) >= self.config.max_client_ids:
            self._client_ids.popitem(last=False)
        self._client_ids[ws_id] = client_id

        # NOTE: Client is added to self.clients AFTER _send_initial_state() in handler()
        # to ensure client receives consistent state before broadcasts

        if is_authenticated:
            self._mark_token_validated(ws_id)

        # Create per-connection message rate limiter
        self._ws_msg_limiters[ws_id] = WebSocketMessageRateLimiter()

        # Register with connection quality tracker
        self._quality_tracker.register(ws_id)

        return (True, client_ip, client_id, ws_id, is_authenticated, ws_token)

    async def _send_initial_state(self, websocket, client_id: str, is_authenticated: bool) -> None:
        """Send initial connection state to client."""
        await websocket.send(
            json.dumps(
                {
                    "type": "connection_info",
                    "data": {
                        "authenticated": is_authenticated,
                        "client_id": client_id[:8] + "...",
                        "write_access": is_authenticated or not auth_config.enabled,
                    },
                }
            )
        )

        await websocket.send(
            json.dumps(
                {
                    "type": "loop_list",
                    "data": {
                        "loops": self.get_loop_list(),
                        "count": len(self.active_loops),
                    },
                }
            )
        )

        # NOTE: We no longer send all cached debate states on connect.
        # This was causing old debate content to appear in new debates.
        # Clients should subscribe to specific debates to receive their state.
        # The loop_list message above tells clients what debates are available.

    async def _send_debate_state(self, websocket, debate_id: str) -> None:
        """Send current debate state to a newly subscribed client.

        Args:
            websocket: The WebSocket connection
            debate_id: The debate ID to send state for
        """
        # Get state under lock
        with self._debate_states_lock:
            state = self.debate_states.get(debate_id)

        if state:
            await websocket.send(
                json.dumps(
                    {
                        "type": "sync",
                        "data": state,
                        "debate_id": debate_id,
                    }
                )
            )
            # If debate is in progress, resend debate_start for UI consistency
            if not state.get("ended"):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "debate_start",
                            "loop_id": debate_id,
                            "data": {
                                "debate_id": debate_id,
                                "status": "in_progress",
                                "task": state.get("task", ""),
                                "agents": state.get("agents", []),
                            },
                        }
                    )
                )
        else:
            # No state yet, send waiting status
            await websocket.send(
                json.dumps(
                    {
                        "type": "sync",
                        "data": {"debate_id": debate_id, "status": "waiting"},
                        "debate_id": debate_id,
                    }
                )
            )

    async def _parse_message(self, message: str) -> tuple[dict | None, str | None]:
        """Parse and validate incoming WebSocket message.

        Returns (parsed_data, error_reason). If parsing succeeds,
        error_reason is None. If it fails, parsed_data is None and
        error_reason contains a categorized description.
        """
        if len(message) > WS_MAX_MESSAGE_SIZE:
            logger.warning("[ws] Message too large from client: %s bytes", len(message))
            return None, "message_too_large"

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, json.loads, message), timeout=5.0
            )
            return result, None
        except asyncio.TimeoutError:
            logger.warning("[ws] JSON parsing timed out - possible DoS attempt")
            return None, "parse_timeout"
        except json.JSONDecodeError as e:
            logger.warning("[ws] Invalid JSON from client: %s", e)
            return None, "invalid_json"
        except RuntimeError as e:
            logger.error("[ws] Event loop error during JSON parsing: %s", e)
            return None, "internal_error"

    async def _handle_user_action(
        self,
        websocket,
        data: dict,
        ws_id: int,
        ws_token: str | None,
        is_authenticated: bool,
        client_id: str,
    ) -> bool:
        """Handle user vote or suggestion message.

        Returns:
            Updated is_authenticated value (may change if token invalidated).
        """
        msg_type = data.get("type")

        # Require authentication for write operations
        if not is_authenticated and auth_config.enabled:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {
                            "message": "Authentication required for voting/suggestions",
                            "code": 401,
                        },
                    }
                )
            )
            return is_authenticated

        # Periodic token revalidation
        if is_authenticated and ws_token and self._should_revalidate_token(ws_id):
            if not auth_config.validate_token(ws_token):
                logger.warning("[ws] Token invalidated for client %s...", client_id[:8])
                await websocket.send(
                    json.dumps(
                        {
                            "type": "auth_revoked",
                            "data": {"message": "Token has been revoked or expired", "code": 401},
                        }
                    )
                )
                return False
            self._mark_token_validated(ws_id)

        stored_client_id = self._client_ids.get(ws_id, secrets.token_urlsafe(16))
        loop_id = data.get("loop_id", "")

        # Validate loop_id
        if not loop_id or loop_id not in self.active_loops:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": f"Invalid or inactive loop_id: {loop_id}"},
                    }
                )
            )
            return is_authenticated

        # Validate token for specific loop_id
        if auth_config.enabled and ws_token:
            is_valid, err_msg = auth_config.validate_token_for_loop(ws_token, loop_id)
            if not is_valid:
                await websocket.send(
                    json.dumps({"type": "error", "data": {"message": err_msg, "code": 403}})
                )
                return is_authenticated

        # Validate payload
        payload, error = self._validate_audience_payload(data)
        if error:
            await websocket.send(json.dumps({"type": "error", "data": {"message": error}}))
            return is_authenticated

        # Rate limiting
        should_cleanup = False
        with self._rate_limiters_lock:
            if stored_client_id not in self._rate_limiters:
                self._rate_limiters[stored_client_id] = TokenBucket(
                    rate_per_minute=10.0, burst_size=5
                )
            self._rate_limiter_last_access[stored_client_id] = time.time()
            rate_limiter = self._rate_limiters[stored_client_id]
            self._rate_limiter_cleanup_counter += 1
            if self._rate_limiter_cleanup_counter >= self._CLEANUP_INTERVAL:
                self._rate_limiter_cleanup_counter = 0
                should_cleanup = True
        if should_cleanup:
            self._cleanup_stale_rate_limiters()

        if not rate_limiter.consume(1):
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": "Rate limited. Please wait before submitting again."},
                    }
                )
            )
            return is_authenticated

        # Process the message
        self._process_audience_message(msg_type, loop_id, payload, stored_client_id)
        await websocket.send(
            json.dumps(
                {"type": "ack", "data": {"message": "Message received", "msg_type": msg_type}}
            )
        )

        return is_authenticated

    async def _handle_wisdom_submission(
        self,
        websocket,
        data: dict,
        ws_id: int,
        ws_token: str | None,
        is_authenticated: bool,
        client_id: str,
    ) -> None:
        """Handle wisdom submission from audience.

        Wisdom submissions are stored and can be injected as fallback
        when AI agents fail to respond.
        """
        # Require authentication if enabled
        if not is_authenticated and auth_config.enabled:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {
                            "message": "Authentication required for wisdom submissions",
                            "code": 401,
                        },
                    }
                )
            )
            return

        loop_id = data.get("loop_id", "")

        # Validate loop_id
        if not loop_id or loop_id not in self.active_loops:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": f"Invalid or inactive loop_id: {loop_id}"},
                    }
                )
            )
            return

        # Extract wisdom text
        text = data.get("text", "").strip()
        if not text or len(text) < 10:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "data": {"message": "Wisdom text must be at least 10 characters"},
                    }
                )
            )
            return

        if len(text) > 280:
            text = text[:280]

        # Get wisdom store
        try:
            from aragora.insights.store import InsightStore

            wisdom_store = InsightStore()

            wisdom_id = wisdom_store.add_wisdom_submission(
                loop_id=loop_id,
                wisdom_data={
                    "text": text,
                    "submitter_id": client_id[:16],
                    "context_tags": data.get("tags", []),
                },
            )

            await websocket.send(
                json.dumps(
                    {
                        "type": "wisdom_confirmed",
                        "data": {"wisdom_id": wisdom_id, "message": "Wisdom received"},
                    }
                )
            )

            logger.info("[wisdom] Stored submission %s for loop %s", wisdom_id, loop_id)

        except (OSError, ValueError, AttributeError, KeyError) as e:
            logger.error("[wisdom] Failed to store submission: %s", e)
            await websocket.send(
                json.dumps(
                    {"type": "error", "data": {"message": "Failed to store wisdom submission"}}
                )
            )

    async def _cleanup_connection(
        self, client_ip: str, client_id: str, ws_id: int, websocket
    ) -> None:
        """Clean up resources after connection closes."""
        async with self._clients_lock:
            self.clients.discard(websocket)
        logger.info(
            "[ws] Client %s... disconnected from %s (remaining_clients=%s)",
            client_id[:8],
            client_ip,
            len(self.clients),
        )

        # SECURITY: Clean up subscription tracking to prevent stale entries
        self._client_subscriptions.pop(ws_id, None)

        stored_client_id = self._client_ids.pop(ws_id, None)
        if stored_client_id:
            with self._rate_limiters_lock:
                self._rate_limiters.pop(stored_client_id, None)
                self._rate_limiter_last_access.pop(stored_client_id, None)

        self._release_ws_connection(client_ip)
        self._ws_token_validated.pop(ws_id, None)
        self._ws_msg_limiters.pop(ws_id, None)
        self._quality_tracker.unregister(ws_id)

    async def handler(self, websocket) -> None:
        """Handle a WebSocket connection with origin validation."""
        # Set up connection with validation
        (
            success,
            client_ip,
            client_id,
            ws_id,
            is_authenticated,
            ws_token,
        ) = await self._setup_connection(websocket)
        if not success:
            return

        logger.info(
            "[ws] Client %s... connected from %s (authenticated=%s, total_clients=%s)",
            client_id[:8],
            client_ip,
            is_authenticated,
            len(self.clients),
        )
        try:
            # Send initial connection state
            await self._send_initial_state(websocket, client_id, is_authenticated)

            # Add client to broadcast set AFTER initial sync completes
            # This prevents receiving broadcasts before state is synchronized
            async with self._clients_lock:
                self.clients.add(websocket)

            # Handle incoming messages
            async for message in websocket:
                # Per-connection message rate limiting
                msg_limiter = self._ws_msg_limiters.get(ws_id)
                if msg_limiter and not msg_limiter.allow_message():
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "data": {
                                    "message": "Message rate limit exceeded. Please slow down.",
                                    "code": 429,
                                },
                            }
                        )
                    )
                    continue

                data, parse_error = await self._parse_message(message)
                if data is None:
                    _error_messages = {
                        "message_too_large": "Message exceeds maximum size limit",
                        "parse_timeout": "Message processing timed out",
                        "invalid_json": "Invalid JSON format",
                        "internal_error": "Server error processing message",
                    }
                    await websocket.send(
                        json.dumps({
                            "type": "error",
                            "data": {
                                "message": _error_messages.get(
                                    parse_error or "", "Invalid message format"
                                ),
                                "error_category": "validation",
                                "error_type": parse_error or "unknown",
                            },
                        })
                    )
                    continue

                msg_type = data.get("type")

                if msg_type == "get_loops":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "loop_list",
                                "data": {
                                    "loops": self.get_loop_list(),
                                    "count": len(self.active_loops),
                                },
                            }
                        )
                    )
                elif msg_type in ("user_vote", "user_suggestion"):
                    is_authenticated = await self._handle_user_action(
                        websocket, data, ws_id, ws_token, is_authenticated, client_id
                    )
                elif msg_type == "wisdom_submission":
                    await self._handle_wisdom_submission(
                        websocket, data, ws_id, ws_token, is_authenticated, client_id
                    )
                elif msg_type == "ping":
                    # Application-level ping for latency measurement.
                    # The client sends {"type": "ping", "ts": <client_timestamp_ms>}
                    # and we echo it back as a pong so the client can compute RTT.
                    client_ts = data.get("ts", 0)
                    server_ts = time.time() * 1000
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "pong",
                                "data": {
                                    "client_ts": client_ts,
                                    "server_ts": server_ts,
                                },
                            }
                        )
                    )
                    # Record server-side latency estimate from client timestamp
                    if client_ts:
                        latency_ms = server_ts - client_ts
                        if 0 <= latency_ms < 60000:  # Sanity: ignore negative or >60s
                            self._quality_tracker.record_latency(ws_id, latency_ms)
                elif msg_type == "connection_quality":
                    # Client can request their connection quality metrics
                    quality = self._quality_tracker.get_quality(ws_id)
                    sub_id = self._client_subscriptions.get(ws_id)
                    buf_metrics = (
                        self._replay_buffer.get_metrics(sub_id)
                        if sub_id
                        else {}
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "connection_quality",
                                "data": {
                                    "client": quality or {},
                                    "buffer": buf_metrics,
                                },
                            }
                        )
                    )
                elif msg_type == "subscribe":
                    # SECURITY: Track client subscription for stream isolation
                    # This ensures clients only receive events for debates they subscribed to
                    debate_id = data.get("debate_id") or data.get("loop_id")
                    if debate_id:
                        # SECURITY: Verify RBAC permission before subscribing
                        subscribe_allowed = True
                        try:
                            from aragora.rbac.checker import check_permission as rbac_check
                            from aragora.rbac.models import AuthorizationContext

                            if is_authenticated and ws_token:
                                # Build auth context from token metadata
                                auth_state = self._ws_auth_states.get(ws_id, {})
                                auth_ctx = AuthorizationContext(
                                    user_id=auth_state.get("user_id", client_id),
                                    roles=auth_state.get("roles", {"viewer"}),
                                )
                                decision = rbac_check(auth_ctx, "debates.read", debate_id)
                                if not decision.allowed:
                                    subscribe_allowed = False
                                    await websocket.send(
                                        json.dumps(
                                            {
                                                "type": "error",
                                                "data": {
                                                    "message": "Permission denied: debates:read required",
                                                    "code": 403,
                                                },
                                            }
                                        )
                                    )
                        except ImportError:
                            pass  # RBAC module not available, allow subscription

                        # Also check tenant isolation
                        if subscribe_allowed:
                            from .tenant_filter import get_tenant_filter

                            tf = get_tenant_filter()
                            tenant_ok, tenant_msg = tf.validate_subscription(ws_id, debate_id)
                            if not tenant_ok:
                                subscribe_allowed = False
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "error",
                                            "data": {
                                                "message": tenant_msg,
                                                "code": 403,
                                            },
                                        }
                                    )
                                )

                        if subscribe_allowed:
                            self._client_subscriptions[ws_id] = debate_id

                            # Check if client is requesting replay (reconnection)
                            replay_from_seq = data.get("replay_from_seq")
                            if replay_from_seq is not None and isinstance(replay_from_seq, (int, float)):
                                replay_from_seq = int(replay_from_seq)
                                self._quality_tracker.record_reconnect(ws_id)
                                logger.info(
                                    "[ws] Client %s... reconnected to %s, replaying from seq %d",
                                    client_id[:8],
                                    debate_id,
                                    replay_from_seq,
                                )
                                # Replay missed events from the buffer
                                missed = self._replay_buffer.replay_since(
                                    debate_id, replay_from_seq
                                )
                                replayed_count = len(missed)
                                self._quality_tracker.record_replay(ws_id, replayed_count)

                                # Send replay_start so client knows replay is beginning
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "replay_start",
                                            "data": {
                                                "debate_id": debate_id,
                                                "from_seq": replay_from_seq,
                                                "event_count": replayed_count,
                                                "buffer_oldest_seq": self._replay_buffer.get_oldest_seq(debate_id),
                                                "buffer_latest_seq": self._replay_buffer.get_latest_seq(debate_id),
                                            },
                                        }
                                    )
                                )

                                # Send replayed events in order
                                for event_json in missed:
                                    try:
                                        await asyncio.wait_for(
                                            websocket.send(event_json), timeout=5.0
                                        )
                                    except (asyncio.TimeoutError, OSError, RuntimeError):
                                        logger.warning(
                                            "[ws] Replay send failed for client %s...",
                                            client_id[:8],
                                        )
                                        break

                                # Send replay_end so client knows replay is complete
                                await websocket.send(
                                    json.dumps(
                                        {
                                            "type": "replay_end",
                                            "data": {
                                                "debate_id": debate_id,
                                                "replayed_count": replayed_count,
                                                "latest_seq": self._replay_buffer.get_latest_seq(debate_id),
                                            },
                                        }
                                    )
                                )
                                logger.info(
                                    "[ws] Replayed %d events to client %s...",
                                    replayed_count,
                                    client_id[:8],
                                )
                            else:
                                # Fresh subscription - send current debate state
                                logger.info(
                                    "[ws] Client %s... subscribed to %s",
                                    client_id[:8],
                                    debate_id,
                                )
                                await self._send_debate_state(websocket, debate_id)
                    else:
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "data": {"message": "subscribe requires debate_id"},
                                }
                            )
                        )

        except Exception as e:  # noqa: BLE001 - Broad catch needed for WebSocket handler robustness
            # WebSocket handlers must catch all exceptions to ensure cleanup runs
            # and to prevent server crashes from individual client errors
            error_name = type(e).__name__
            if "ConnectionClosed" not in error_name and "ConnectionClosedOK" not in error_name:
                logger.error(
                    "[ws] Unexpected error for client %s...: %s: %s", client_id[:8], error_name, e
                )
        finally:
            await self._cleanup_connection(client_ip, client_id, ws_id, websocket)

    def start_tts_bridge(self, event_bus: Any, tts_integration: Any, voice_handler: Any) -> None:
        """Create and connect a TTS event bridge for live voice synthesis.

        Call this after enabling ``enable_tts`` when a voice session becomes
        active.  The bridge subscribes to ``agent_message`` events on the
        supplied event bus and automatically synthesizes TTS audio through the
        voice handler.

        Args:
            event_bus: Debate :class:`EventBus` instance.
            tts_integration: :class:`TTSIntegration` instance.
            voice_handler: :class:`VoiceStreamHandler` instance.
        """
        if not self.enable_tts:
            logger.debug("[ws] TTS bridge not started (enable_tts=False)")
            return

        try:
            from aragora.server.stream.tts_event_bridge import TTSEventBridge

            self._tts_bridge = TTSEventBridge(
                tts=tts_integration,
                voice_handler=voice_handler,
            )
            self._tts_bridge.connect(event_bus)
            logger.info("[ws] TTS event bridge started")
        except (ImportError, RuntimeError, TypeError) as e:
            logger.warning("[ws] Failed to start TTS event bridge: %s", e)

    async def stop_tts_bridge(self) -> None:
        """Shut down the TTS event bridge if running."""
        if self._tts_bridge is not None:
            try:
                await self._tts_bridge.shutdown()
            except (RuntimeError, OSError) as e:
                logger.warning("[ws] Error stopping TTS bridge: %s", e)
            finally:
                self._tts_bridge = None
            logger.info("[ws] TTS event bridge stopped")

    async def _buffer_cleanup_loop(self) -> None:
        """Periodically remove replay buffers for debates that are no longer active."""
        while self._running:
            await asyncio.sleep(60)
            if not self._running:
                break
            try:
                with self._active_loops_lock:
                    active_ids = set(self.active_loops.keys())
                removed = self._replay_buffer.cleanup_stale(active_ids)
                if removed:
                    logger.debug("[ws] Cleaned up %d stale replay buffers", removed)
            except (RuntimeError, AttributeError) as e:
                logger.debug("[ws] Buffer cleanup error: %s", e)

    async def start(self) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self._running = True
        # Create stop event for graceful shutdown
        self._stop_event = asyncio.Event()
        # Store task reference and add error callback to prevent silent failures
        self._drain_task = asyncio.create_task(self._drain_loop())
        self._drain_task.add_done_callback(self._handle_drain_task_error)
        # Start application-level heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # Start replay buffer cleanup loop
        self._cleanup_task = asyncio.create_task(self._buffer_cleanup_loop())

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            max_size=WS_MAX_MESSAGE_SIZE,
            ping_interval=30,  # Send ping every 30s
            ping_timeout=10,  # Close connection if no pong within 10s
            compression="deflate",  # Enable permessage-deflate for reduced bandwidth
        ):
            logger.info(
                "WebSocket server: ws://%s:%s (max message size: %s bytes)",
                self.host,
                self.port,
                WS_MAX_MESSAGE_SIZE,
            )
            await self._stop_event.wait()  # Run until shutdown signal

    def stop(self) -> None:
        """Stop the server."""
        self._running = False

    def _handle_drain_task_error(self, task: asyncio.Task) -> None:
        """Handle errors from the drain loop task."""
        try:
            exc = task.exception()
            if exc is not None:
                logger.error("Drain loop task failed with exception: %s", exc)
        except asyncio.CancelledError:
            pass  # Task was cancelled, not an error

    async def graceful_shutdown(self) -> None:
        """Gracefully close all client connections and stop the server."""
        self._running = False
        # Shut down TTS bridge before closing connections
        await self.stop_tts_bridge()
        # Signal the server to stop
        if self._stop_event:
            self._stop_event.set()
        # Close all connected clients under lock
        async with self._clients_lock:
            if self.clients:
                close_tasks = []
                for client in list(self.clients):
                    try:
                        close_tasks.append(client.close())
                    except (OSError, RuntimeError) as e:
                        logger.debug("Error closing WebSocket client: %s", e)
                if close_tasks:
                    await asyncio.gather(*close_tasks, return_exceptions=True)
                self.clients.clear()


# ---------------------------------------------------------------------------
# Module-level replay buffer accessor (for HTTP polling endpoint)
# ---------------------------------------------------------------------------

_global_replay_buffer: EventReplayBuffer | None = None


def get_global_replay_buffer() -> EventReplayBuffer | None:
    """Return the module-level replay buffer singleton, if set.

    This is set automatically when a DebateStreamServer is created,
    allowing the HTTP polling endpoint to access the same replay data.
    """
    return _global_replay_buffer


def set_global_replay_buffer(buffer: EventReplayBuffer | None) -> None:
    """Set the module-level replay buffer singleton."""
    global _global_replay_buffer
    _global_replay_buffer = buffer


__all__ = [
    "ConnectionQualityTracker",
    "DebateStreamServer",
    "EventReplayBuffer",
    "get_global_replay_buffer",
    "set_global_replay_buffer",
]
