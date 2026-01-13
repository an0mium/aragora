"""
WebSocket broadcasting and client management utilities.

Extracted from servers.py to provide reusable broadcast functionality
for both DebateStreamServer and AiohttpUnifiedServer.
"""

import asyncio
import json
import logging
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Set

from .emitter import AudienceInbox, SyncEventEmitter, TokenBucket
from .events import StreamEvent, StreamEventType
from .state_manager import LoopInstance

logger = logging.getLogger(__name__)


@dataclass
class BroadcasterConfig:
    """Configuration for broadcaster behavior."""

    # Loop tracking limits
    max_active_loops: int = 1000
    active_loops_ttl: int = 86400  # 24 hours

    # Debate state limits
    max_debate_states: int = 500
    debate_states_ttl: int = 3600  # 1 hour

    # Rate limiting
    rate_limiter_ttl: int = 3600  # 1 hour
    cleanup_interval: int = 100  # Cleanup every N accesses
    rate_limit_per_minute: float = 60.0  # Tokens per minute per client
    rate_limit_burst: int = 10  # Max burst size

    # Client tracking
    max_client_ids: int = 10000


class ClientManager:
    """
    Manages WebSocket client connections with secure ID mapping and rate limiting.

    Provides:
    - Secure client ID generation (cryptographically random)
    - LRU eviction for client ID mapping
    - Per-client rate limiting with TTL cleanup

    Thread-safe for concurrent access.
    """

    def __init__(self, config: Optional[BroadcasterConfig] = None):
        self.config = config or BroadcasterConfig()

        # Connected clients (websocket objects)
        self.clients: Set[Any] = set()
        self._clients_lock = threading.Lock()

        # Secure client ID mapping: websocket_id -> client_id
        self._client_ids: OrderedDict[int, str] = OrderedDict()

        # Rate limiters per client
        self._rate_limiters: dict[str, TokenBucket] = {}
        self._rate_limiter_last_access: dict[str, float] = {}
        self._rate_limiters_lock = threading.Lock()
        self._cleanup_counter = 0

    def add_client(self, websocket: Any) -> str:
        """Add a client and return its secure ID.

        Args:
            websocket: WebSocket connection object

        Returns:
            Cryptographically secure client ID
        """
        ws_id = id(websocket)

        with self._clients_lock:
            self.clients.add(websocket)

            # Generate secure client ID if not already tracked
            if ws_id not in self._client_ids:
                # LRU eviction if at capacity
                if len(self._client_ids) >= self.config.max_client_ids:
                    self._client_ids.popitem(last=False)

                client_id = secrets.token_urlsafe(16)
                self._client_ids[ws_id] = client_id
            else:
                # Move to end for LRU
                self._client_ids.move_to_end(ws_id)
                client_id = self._client_ids[ws_id]

        return client_id

    def remove_client(self, websocket: Any) -> None:
        """Remove a client connection."""
        ws_id = id(websocket)

        with self._clients_lock:
            self.clients.discard(websocket)
            self._client_ids.pop(ws_id, None)

    def get_client_id(self, websocket: Any) -> Optional[str]:
        """Get the secure client ID for a websocket."""
        ws_id = id(websocket)
        return self._client_ids.get(ws_id)

    def get_rate_limiter(self, client_id: str) -> TokenBucket:
        """Get or create rate limiter for a client.

        Args:
            client_id: Secure client identifier

        Returns:
            TokenBucket for this client
        """
        # Periodic cleanup
        self._cleanup_counter += 1
        if self._cleanup_counter >= self.config.cleanup_interval:
            self._cleanup_counter = 0
            self._cleanup_stale_rate_limiters()

        with self._rate_limiters_lock:
            if client_id not in self._rate_limiters:
                self._rate_limiters[client_id] = TokenBucket(
                    rate_per_minute=self.config.rate_limit_per_minute,
                    burst_size=self.config.rate_limit_burst,
                )
            self._rate_limiter_last_access[client_id] = time.time()
            return self._rate_limiters[client_id]

    def _cleanup_stale_rate_limiters(self) -> None:
        """Remove rate limiters not accessed within TTL."""
        now = time.time()
        with self._rate_limiters_lock:
            stale_keys = [
                k
                for k, v in self._rate_limiter_last_access.items()
                if now - v > self.config.rate_limiter_ttl
            ]
            for k in stale_keys:
                self._rate_limiters.pop(k, None)
                self._rate_limiter_last_access.pop(k, None)
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limiters")

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self.clients)

    # Context manager support for proper cleanup
    def __enter__(self) -> "ClientManager":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - cleanup all resources."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up all tracked resources."""
        with self._clients_lock:
            self.clients.clear()
            self._client_ids.clear()

        with self._rate_limiters_lock:
            self._rate_limiters.clear()
            self._rate_limiter_last_access.clear()

        logger.debug("ClientManager resources cleaned up")


class DebateStateCache:
    """
    Caches debate state for late-joiner synchronization.

    Maintains a bounded cache of debate states with TTL-based eviction
    for ended debates.

    Thread-safe for concurrent access.
    """

    def __init__(self, config: Optional[BroadcasterConfig] = None):
        self.config = config or BroadcasterConfig()

        self.debate_states: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._last_access: dict[str, float] = {}

    def update_from_event(self, event: StreamEvent) -> None:
        """Update debate state based on emitted event.

        Args:
            event: Stream event to process
        """
        loop_id = event.loop_id

        with self._lock:
            if event.type == StreamEventType.DEBATE_START:
                self._handle_debate_start(loop_id, event)
            elif event.type == StreamEventType.AGENT_MESSAGE:
                self._handle_agent_message(loop_id, event)
            elif event.type == StreamEventType.ROUND_START:
                self._handle_round_start(loop_id, event)
            elif event.type == StreamEventType.CONSENSUS:
                self._handle_consensus(loop_id, event)
            elif event.type == StreamEventType.DEBATE_END:
                self._handle_debate_end(loop_id, event)
            elif event.type == StreamEventType.LOOP_UNREGISTER:
                self._handle_loop_unregister(loop_id)

    def _handle_debate_start(self, loop_id: str, event: StreamEvent) -> None:
        """Initialize state for new debate."""
        # Enforce max size with LRU eviction (prefer ended debates)
        if len(self.debate_states) >= self.config.max_debate_states:
            ended_states = [
                (k, self._last_access.get(k, 0))
                for k, v in self.debate_states.items()
                if v.get("ended")
            ]
            if ended_states:
                oldest = min(ended_states, key=lambda x: x[1])[0]
                self.debate_states.pop(oldest, None)
                self._last_access.pop(oldest, None)

        self.debate_states[loop_id] = {
            "id": loop_id,
            "task": event.data.get("task"),
            "agents": event.data.get("agents"),
            "messages": [],
            "consensus_reached": False,
            "consensus_confidence": 0.0,
            "consensus_answer": "",
            "started_at": event.timestamp,
            "rounds": 0,
            "ended": False,
            "duration": 0.0,
        }
        self._last_access[loop_id] = time.time()

    def _handle_agent_message(self, loop_id: str, event: StreamEvent) -> None:
        """Add message to debate state."""
        if loop_id not in self.debate_states:
            return

        state = self.debate_states[loop_id]
        state["messages"].append(
            {
                "agent": event.agent,
                "role": event.data.get("role"),
                "round": event.round,
                "content": event.data.get("content"),
            }
        )
        # Cap at last 1000 messages
        if len(state["messages"]) > 1000:
            state["messages"] = state["messages"][-1000:]
        self._last_access[loop_id] = time.time()

    def _handle_round_start(self, loop_id: str, event: StreamEvent) -> None:
        """Update round count."""
        if loop_id in self.debate_states:
            self.debate_states[loop_id]["rounds"] = event.round
            self._last_access[loop_id] = time.time()

    def _handle_consensus(self, loop_id: str, event: StreamEvent) -> None:
        """Update consensus info."""
        if loop_id not in self.debate_states:
            return

        state = self.debate_states[loop_id]
        state["consensus_reached"] = event.data.get("reached", False)
        state["consensus_confidence"] = event.data.get("confidence", 0.0)
        state["consensus_answer"] = event.data.get("answer", "")
        self._last_access[loop_id] = time.time()

    def _handle_debate_end(self, loop_id: str, event: StreamEvent) -> None:
        """Mark debate as ended."""
        if loop_id in self.debate_states:
            state = self.debate_states[loop_id]
            state["ended"] = True
            state["duration"] = event.data.get("duration", 0.0)
            self._last_access[loop_id] = time.time()

    def _handle_loop_unregister(self, loop_id: str) -> None:
        """Remove state when loop unregisters."""
        self.debate_states.pop(loop_id, None)
        self._last_access.pop(loop_id, None)

    def get_state(self, loop_id: str) -> Optional[dict]:
        """Get cached state for a debate."""
        with self._lock:
            if loop_id in self.debate_states:
                self._last_access[loop_id] = time.time()
                return self.debate_states[loop_id].copy()
        return None

    def cleanup_stale(self) -> int:
        """Remove stale ended debates. Returns count of removed entries."""
        now = time.time()
        removed = 0
        with self._lock:
            stale_keys = [
                k
                for k, v in self.debate_states.items()
                if v.get("ended")
                and now - self._last_access.get(k, 0) > self.config.debate_states_ttl
            ]
            for k in stale_keys:
                self.debate_states.pop(k, None)
                self._last_access.pop(k, None)
                removed += 1
        return removed

    # Context manager support
    def __enter__(self) -> "DebateStateCache":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - cleanup all resources."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up all tracked resources."""
        with self._lock:
            self.debate_states.clear()
            self._last_access.clear()
        logger.debug("DebateStateCache resources cleaned up")


class LoopRegistry:
    """
    Tracks active nomic loop instances.

    Provides registration, unregistration, and state updates for loops
    with TTL-based cleanup of stale entries.

    Thread-safe for concurrent access.
    """

    def __init__(self, config: Optional[BroadcasterConfig] = None):
        self.config = config or BroadcasterConfig()

        self.active_loops: dict[str, LoopInstance] = {}
        self._lock = threading.Lock()
        self._last_access: dict[str, float] = {}

    def register(self, loop_id: str, name: str, path: str = "") -> LoopInstance:
        """Register a new loop instance.

        Args:
            loop_id: Unique identifier for the loop
            name: Human-readable name
            path: Optional path/location

        Returns:
            The created LoopInstance
        """
        instance = LoopInstance(
            loop_id=loop_id,
            name=name,
            started_at=time.time(),
            path=path,
        )

        with self._lock:
            # LRU eviction if at capacity
            if len(self.active_loops) >= self.config.max_active_loops:
                oldest = min(self._last_access, key=self._last_access.get, default=None)
                if oldest:
                    self.active_loops.pop(oldest, None)
                    self._last_access.pop(oldest, None)

            self.active_loops[loop_id] = instance
            self._last_access[loop_id] = time.time()

        return instance

    def unregister(self, loop_id: str) -> bool:
        """Unregister a loop instance.

        Returns:
            True if loop was found and removed
        """
        with self._lock:
            if loop_id in self.active_loops:
                del self.active_loops[loop_id]
                self._last_access.pop(loop_id, None)
                return True
        return False

    def update_state(
        self, loop_id: str, cycle: Optional[int] = None, phase: Optional[str] = None
    ) -> bool:
        """Update loop state.

        Returns:
            True if loop exists and was updated
        """
        with self._lock:
            if loop_id not in self.active_loops:
                return False
            if cycle is not None:
                self.active_loops[loop_id].cycle = cycle
            if phase is not None:
                self.active_loops[loop_id].phase = phase
            self._last_access[loop_id] = time.time()
        return True

    def get_list(self) -> list[dict]:
        """Get list of active loops for client sync."""
        with self._lock:
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

    def get(self, loop_id: str) -> Optional[LoopInstance]:
        """Get a specific loop instance."""
        with self._lock:
            if loop_id in self.active_loops:
                self._last_access[loop_id] = time.time()
                return self.active_loops[loop_id]
        return None

    @property
    def count(self) -> int:
        """Number of active loops."""
        return len(self.active_loops)

    def cleanup_stale(self) -> int:
        """Remove stale loops. Returns count of removed entries."""
        now = time.time()
        removed = 0
        with self._lock:
            stale_keys = [
                k for k, v in self._last_access.items() if now - v > self.config.active_loops_ttl
            ]
            for k in stale_keys:
                self.active_loops.pop(k, None)
                self._last_access.pop(k, None)
                removed += 1
        return removed

    # Context manager support
    def __enter__(self) -> "LoopRegistry":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - cleanup all resources."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up all tracked resources."""
        with self._lock:
            self.active_loops.clear()
            self._last_access.clear()
        logger.debug("LoopRegistry resources cleaned up")


class WebSocketBroadcaster:
    """
    Core broadcasting functionality for WebSocket servers.

    Combines ClientManager, DebateStateCache, and LoopRegistry to provide
    a complete broadcasting solution. Uses composition to allow servers
    to use individual components or the full broadcaster.

    Usage:
        broadcaster = WebSocketBroadcaster()

        # In connection handler:
        client_id = broadcaster.client_manager.add_client(websocket)

        # To broadcast:
        await broadcaster.broadcast(event)

        # To start drain loop:
        asyncio.create_task(broadcaster.drain_loop())
    """

    def __init__(
        self,
        emitter: Optional[SyncEventEmitter] = None,
        config: Optional[BroadcasterConfig] = None,
    ):
        self.config = config or BroadcasterConfig()

        # Core components
        self.client_manager = ClientManager(self.config)
        self.debate_state_cache = DebateStateCache(self.config)
        self.loop_registry = LoopRegistry(self.config)
        self.audience_inbox = AudienceInbox()

        # Event emitter
        self._emitter = emitter or SyncEventEmitter()
        self._running = False

        # Subscribe to emitter for state updates
        self._emitter.subscribe(self.debate_state_cache.update_from_event)

    @property
    def emitter(self) -> SyncEventEmitter:
        """Get the event emitter."""
        return self._emitter

    @property
    def clients(self) -> Set[Any]:
        """Get connected clients (for backward compatibility)."""
        return self.client_manager.clients

    async def broadcast(self, event: StreamEvent) -> None:
        """Send event to all connected clients.

        Args:
            event: Event to broadcast
        """
        clients = self.client_manager.clients
        if not clients:
            return

        message = event.to_json()
        disconnected = set()

        for client in clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.debug(f"Client disconnected during broadcast: {e}")
                disconnected.add(client)

        # Remove disconnected clients
        for client in disconnected:
            self.client_manager.remove_client(client)

    async def broadcast_batch(self, events: list[StreamEvent]) -> None:
        """Send multiple events in a single message.

        Batching reduces WebSocket overhead by sending events as a JSON array.

        Args:
            events: List of events to broadcast
        """
        clients = self.client_manager.clients
        if not clients or not events:
            return

        message = json.dumps([e.to_dict() for e in events])
        disconnected = set()

        for client in clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.debug(f"Client disconnected during batch broadcast: {e}")
                disconnected.add(client)

        for client in disconnected:
            self.client_manager.remove_client(client)

    async def drain_loop(self) -> None:
        """Background task that drains emitter queue and broadcasts.

        Uses batching to reduce WebSocket overhead. Should be started
        as an asyncio task when the server starts.
        """
        self._running = True
        while self._running:
            events = list(self._emitter.drain())
            if events:
                await self.broadcast_batch(events)
            await asyncio.sleep(0.05)

    def stop(self) -> None:
        """Stop the drain loop."""
        self._running = False

    def cleanup_all(self) -> dict:
        """Run cleanup on all components.

        Returns:
            Dict with counts of cleaned items per component
        """
        return {
            "rate_limiters": self.client_manager._cleanup_stale_rate_limiters() or 0,  # type: ignore[attr-defined, func-returns-value]
            "debate_states": self.debate_state_cache.cleanup_stale(),
            "loops": self.loop_registry.cleanup_stale(),
        }

    # Context manager support
    def __enter__(self) -> "WebSocketBroadcaster":
        """Enter context manager."""
        self._running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - cleanup all resources."""
        self.cleanup()

    async def __aenter__(self) -> "WebSocketBroadcaster":
        """Enter async context manager."""
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager - cleanup all resources."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up all resources across all components."""
        self._running = False
        self.client_manager.cleanup()
        self.debate_state_cache.cleanup()
        self.loop_registry.cleanup()
        logger.debug("WebSocketBroadcaster resources cleaned up")


__all__ = [
    "BroadcasterConfig",
    "ClientManager",
    "DebateStateCache",
    "LoopRegistry",
    "WebSocketBroadcaster",
]
