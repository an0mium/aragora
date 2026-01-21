"""
Redis-backed session store for horizontal scaling.

Provides distributed state management for WebSocket servers:
- Debate states (cached debate progress)
- Active loops (running nomic loop instances)
- WebSocket auth states (authenticated connections)
- Rate limiter state (per-client rate limiting)

When Redis is available, state is shared across server instances.
Falls back to in-memory storage when Redis is unavailable.

Usage:
    from aragora.server.session_store import get_session_store

    store = get_session_store()

    # Store debate state
    store.set_debate_state("loop-123", {"status": "running", ...})
    state = store.get_debate_state("loop-123")

    # Check if using Redis
    if store.is_distributed:
        print("State shared across instances")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from aragora.control_plane.leader import (
    is_distributed_state_required,
    DistributedStateError,
)

logger = logging.getLogger(__name__)

# Configurable session store limits via environment variables
_SESSION_DEBATE_STATE_TTL = int(os.getenv("ARAGORA_SESSION_DEBATE_TTL", "3600"))
_SESSION_ACTIVE_LOOP_TTL = int(os.getenv("ARAGORA_SESSION_LOOP_TTL", "86400"))
_SESSION_AUTH_STATE_TTL = int(os.getenv("ARAGORA_SESSION_AUTH_TTL", "3600"))
_SESSION_RATE_LIMITER_TTL = int(os.getenv("ARAGORA_SESSION_RATE_LIMIT_TTL", "300"))
_SESSION_MAX_DEBATE_STATES = int(os.getenv("ARAGORA_SESSION_MAX_DEBATES", "500"))
_SESSION_MAX_ACTIVE_LOOPS = int(os.getenv("ARAGORA_SESSION_MAX_LOOPS", "1000"))
_SESSION_MAX_AUTH_STATES = int(os.getenv("ARAGORA_SESSION_MAX_AUTH", "10000"))


@dataclass
class SessionStoreConfig:
    """Configuration for session store.

    All values configurable via environment variables:
    - ARAGORA_SESSION_DEBATE_TTL: Debate state TTL in seconds (default 3600)
    - ARAGORA_SESSION_LOOP_TTL: Active loop TTL in seconds (default 86400)
    - ARAGORA_SESSION_AUTH_TTL: Auth state TTL in seconds (default 3600)
    - ARAGORA_SESSION_RATE_LIMIT_TTL: Rate limiter TTL in seconds (default 300)
    - ARAGORA_SESSION_MAX_DEBATES: Max debate states in memory (default 500)
    - ARAGORA_SESSION_MAX_LOOPS: Max active loops in memory (default 1000)
    - ARAGORA_SESSION_MAX_AUTH: Max auth states in memory (default 10000)
    """

    # TTL settings (in seconds)
    debate_state_ttl: int = field(default=_SESSION_DEBATE_STATE_TTL)
    active_loop_ttl: int = field(default=_SESSION_ACTIVE_LOOP_TTL)
    auth_state_ttl: int = field(default=_SESSION_AUTH_STATE_TTL)
    rate_limiter_ttl: int = field(default=_SESSION_RATE_LIMITER_TTL)

    # Max entries (for in-memory store)
    max_debate_states: int = field(default=_SESSION_MAX_DEBATE_STATES)
    max_active_loops: int = field(default=_SESSION_MAX_ACTIVE_LOOPS)
    max_auth_states: int = field(default=_SESSION_MAX_AUTH_STATES)

    # Redis key prefix
    key_prefix: str = "aragora:session:"


class SessionStore(ABC):
    """Abstract base class for session stores."""

    @property
    @abstractmethod
    def is_distributed(self) -> bool:
        """Return True if store is distributed (Redis)."""
        pass

    # Debate state methods
    @abstractmethod
    def get_debate_state(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """Get cached debate state."""
        pass

    @abstractmethod
    def set_debate_state(self, loop_id: str, state: Dict[str, Any]) -> None:
        """Set debate state."""
        pass

    @abstractmethod
    def delete_debate_state(self, loop_id: str) -> bool:
        """Delete debate state. Returns True if deleted."""
        pass

    # Active loop methods
    @abstractmethod
    def get_active_loop(self, loop_id: str) -> Optional[Dict[str, Any]]:
        """Get active loop data."""
        pass

    @abstractmethod
    def set_active_loop(self, loop_id: str, data: Dict[str, Any]) -> None:
        """Set active loop data."""
        pass

    @abstractmethod
    def delete_active_loop(self, loop_id: str) -> bool:
        """Delete active loop. Returns True if deleted."""
        pass

    @abstractmethod
    def list_active_loops(self) -> list[str]:
        """List all active loop IDs."""
        pass

    # Auth state methods
    @abstractmethod
    def get_auth_state(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get WebSocket auth state."""
        pass

    @abstractmethod
    def set_auth_state(self, connection_id: str, state: Dict[str, Any]) -> None:
        """Set WebSocket auth state."""
        pass

    @abstractmethod
    def delete_auth_state(self, connection_id: str) -> bool:
        """Delete auth state. Returns True if deleted."""
        pass

    # Pub/Sub for cross-server messaging
    @abstractmethod
    def publish_event(self, channel: str, event: Dict[str, Any]) -> None:
        """Publish event to channel for cross-server messaging."""
        pass

    @abstractmethod
    def subscribe_events(self, channel: str, callback) -> None:
        """Subscribe to events on channel."""
        pass

    # Cleanup
    @abstractmethod
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired entries. Returns counts by type."""
        pass


class InMemorySessionStore(SessionStore):
    """In-memory session store for single-server deployment."""

    def __init__(self, config: Optional[SessionStoreConfig] = None):
        self._config = config or SessionStoreConfig()

        # State storage with last-access tracking
        self._debate_states: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._debate_states_access: Dict[str, float] = {}
        self._debate_lock = threading.Lock()

        self._active_loops: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._active_loops_access: Dict[str, float] = {}
        self._loops_lock = threading.Lock()

        self._auth_states: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._auth_states_access: Dict[str, float] = {}
        self._auth_lock = threading.Lock()

        # Event subscribers (for local pub/sub)
        self._subscribers: Dict[str, list] = {}
        self._sub_lock = threading.Lock()

    @property
    def is_distributed(self) -> bool:
        return False

    # Debate state methods
    def get_debate_state(self, loop_id: str) -> Optional[Dict[str, Any]]:
        with self._debate_lock:
            if loop_id in self._debate_states:
                self._debate_states_access[loop_id] = time.time()
                self._debate_states.move_to_end(loop_id)
                return self._debate_states[loop_id].copy()
            return None

    def set_debate_state(self, loop_id: str, state: Dict[str, Any]) -> None:
        with self._debate_lock:
            # Enforce max limit with LRU eviction
            while len(self._debate_states) >= self._config.max_debate_states:
                oldest = next(iter(self._debate_states))
                del self._debate_states[oldest]
                self._debate_states_access.pop(oldest, None)

            self._debate_states[loop_id] = state.copy()
            self._debate_states_access[loop_id] = time.time()
            self._debate_states.move_to_end(loop_id)

    def delete_debate_state(self, loop_id: str) -> bool:
        with self._debate_lock:
            if loop_id in self._debate_states:
                del self._debate_states[loop_id]
                self._debate_states_access.pop(loop_id, None)
                return True
            return False

    # Active loop methods
    def get_active_loop(self, loop_id: str) -> Optional[Dict[str, Any]]:
        with self._loops_lock:
            if loop_id in self._active_loops:
                self._active_loops_access[loop_id] = time.time()
                self._active_loops.move_to_end(loop_id)
                return self._active_loops[loop_id].copy()
            return None

    def set_active_loop(self, loop_id: str, data: Dict[str, Any]) -> None:
        with self._loops_lock:
            while len(self._active_loops) >= self._config.max_active_loops:
                oldest = next(iter(self._active_loops))
                del self._active_loops[oldest]
                self._active_loops_access.pop(oldest, None)

            self._active_loops[loop_id] = data.copy()
            self._active_loops_access[loop_id] = time.time()
            self._active_loops.move_to_end(loop_id)

    def delete_active_loop(self, loop_id: str) -> bool:
        with self._loops_lock:
            if loop_id in self._active_loops:
                del self._active_loops[loop_id]
                self._active_loops_access.pop(loop_id, None)
                return True
            return False

    def list_active_loops(self) -> list[str]:
        with self._loops_lock:
            return list(self._active_loops.keys())

    # Auth state methods
    def get_auth_state(self, connection_id: str) -> Optional[Dict[str, Any]]:
        with self._auth_lock:
            if connection_id in self._auth_states:
                self._auth_states_access[connection_id] = time.time()
                return self._auth_states[connection_id].copy()
            return None

    def set_auth_state(self, connection_id: str, state: Dict[str, Any]) -> None:
        with self._auth_lock:
            while len(self._auth_states) >= self._config.max_auth_states:
                oldest = next(iter(self._auth_states))
                del self._auth_states[oldest]
                self._auth_states_access.pop(oldest, None)

            self._auth_states[connection_id] = state.copy()
            self._auth_states_access[connection_id] = time.time()

    def delete_auth_state(self, connection_id: str) -> bool:
        with self._auth_lock:
            if connection_id in self._auth_states:
                del self._auth_states[connection_id]
                self._auth_states_access.pop(connection_id, None)
                return True
            return False

    # Pub/Sub (local only for in-memory store)
    def publish_event(self, channel: str, event: Dict[str, Any]) -> None:
        with self._sub_lock:
            callbacks = self._subscribers.get(channel, [])
        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")

    def subscribe_events(self, channel: str, callback) -> None:
        with self._sub_lock:
            if channel not in self._subscribers:
                self._subscribers[channel] = []
            self._subscribers[channel].append(callback)

    # Cleanup
    def cleanup_expired(self) -> Dict[str, int]:
        now = time.time()
        counts = {"debate_states": 0, "active_loops": 0, "auth_states": 0}

        # Cleanup debate states
        cutoff = now - self._config.debate_state_ttl
        with self._debate_lock:
            expired = [k for k, v in self._debate_states_access.items() if v < cutoff]
            for k in expired:
                self._debate_states.pop(k, None)
                self._debate_states_access.pop(k, None)
            counts["debate_states"] = len(expired)

        # Cleanup active loops
        cutoff = now - self._config.active_loop_ttl
        with self._loops_lock:
            expired = [k for k, v in self._active_loops_access.items() if v < cutoff]
            for k in expired:
                self._active_loops.pop(k, None)
                self._active_loops_access.pop(k, None)
            counts["active_loops"] = len(expired)

        # Cleanup auth states
        cutoff = now - self._config.auth_state_ttl
        with self._auth_lock:
            expired = [k for k, v in self._auth_states_access.items() if v < cutoff]
            for k in expired:
                self._auth_states.pop(k, None)
                self._auth_states_access.pop(k, None)
            counts["auth_states"] = len(expired)

        return counts


class RedisSessionStore(SessionStore):
    """Redis-backed session store for horizontal scaling."""

    def __init__(self, config: Optional[SessionStoreConfig] = None):
        self._config = config or SessionStoreConfig()
        self._prefix = self._config.key_prefix

        # Get Redis client
        from aragora.server.redis_config import get_redis_client

        self._redis = get_redis_client()
        if self._redis is None:
            raise RuntimeError("Redis not available")

        # Pub/Sub subscriber thread
        self._pubsub = None
        self._pubsub_thread = None
        self._callbacks: Dict[str, list] = {}
        self._callbacks_lock = threading.Lock()

    @property
    def is_distributed(self) -> bool:
        return True

    def _key(self, *parts: str) -> str:
        """Build Redis key with prefix."""
        return self._prefix + ":".join(parts)

    # Debate state methods
    def get_debate_state(self, loop_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = self._redis.get(self._key("debate", loop_id))
            if data:
                # Refresh TTL on access
                self._redis.expire(self._key("debate", loop_id), self._config.debate_state_ttl)
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get_debate_state error: {e}")
            return None

    def set_debate_state(self, loop_id: str, state: Dict[str, Any]) -> None:
        try:
            self._redis.setex(
                self._key("debate", loop_id),
                self._config.debate_state_ttl,
                json.dumps(state, default=str),
            )
        except Exception as e:
            logger.warning(f"Redis set_debate_state error: {e}")

    def delete_debate_state(self, loop_id: str) -> bool:
        try:
            return self._redis.delete(self._key("debate", loop_id)) > 0
        except Exception as e:
            logger.warning(f"Redis delete_debate_state error: {e}")
            return False

    # Active loop methods
    # NOTE: Using individual keys with TTL instead of hash to support per-entry expiration
    def get_active_loop(self, loop_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = self._redis.get(self._key("loop", loop_id))
            if data:
                # Refresh TTL on access
                self._redis.expire(self._key("loop", loop_id), self._config.active_loop_ttl)
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get_active_loop error: {e}")
            return None

    def set_active_loop(self, loop_id: str, data: Dict[str, Any]) -> None:
        try:
            # Use setex with TTL for automatic expiration
            self._redis.setex(
                self._key("loop", loop_id),
                self._config.active_loop_ttl,
                json.dumps(data, default=str),
            )
            # Also add to index set (for list_active_loops)
            self._redis.sadd(self._key("loop_index"), loop_id)
        except Exception as e:
            logger.warning(f"Redis set_active_loop error: {e}")

    def delete_active_loop(self, loop_id: str) -> bool:
        try:
            deleted = self._redis.delete(self._key("loop", loop_id)) > 0
            # Remove from index set
            self._redis.srem(self._key("loop_index"), loop_id)
            return deleted
        except Exception as e:
            logger.warning(f"Redis delete_active_loop error: {e}")
            return False

    def list_active_loops(self) -> list[str]:
        """List all active loop IDs.

        Uses an index set for efficiency. Cleans up stale entries.
        """
        try:
            # Get all IDs from index
            all_ids = self._redis.smembers(self._key("loop_index"))
            if not all_ids:
                return []

            # Filter to only those that still exist (haven't expired)
            active_ids = []
            stale_ids = []

            for loop_id in all_ids:
                # Decode bytes if needed
                if isinstance(loop_id, bytes):
                    loop_id = loop_id.decode()

                # Check if key still exists
                if self._redis.exists(self._key("loop", loop_id)):
                    active_ids.append(loop_id)
                else:
                    stale_ids.append(loop_id)

            # Clean up stale index entries
            if stale_ids:
                self._redis.srem(self._key("loop_index"), *stale_ids)
                logger.debug(f"Cleaned up {len(stale_ids)} stale loop index entries")

            return active_ids
        except Exception as e:
            logger.warning(f"Redis list_active_loops error: {e}")
            return []

    # Auth state methods
    def get_auth_state(self, connection_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = self._redis.get(self._key("auth", connection_id))
            if data:
                self._redis.expire(self._key("auth", connection_id), self._config.auth_state_ttl)
                return json.loads(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get_auth_state error: {e}")
            return None

    def set_auth_state(self, connection_id: str, state: Dict[str, Any]) -> None:
        try:
            self._redis.setex(
                self._key("auth", connection_id),
                self._config.auth_state_ttl,
                json.dumps(state, default=str),
            )
        except Exception as e:
            logger.warning(f"Redis set_auth_state error: {e}")

    def delete_auth_state(self, connection_id: str) -> bool:
        try:
            return self._redis.delete(self._key("auth", connection_id)) > 0
        except Exception as e:
            logger.warning(f"Redis delete_auth_state error: {e}")
            return False

    # Pub/Sub for cross-server messaging
    def publish_event(self, channel: str, event: Dict[str, Any]) -> None:
        try:
            self._redis.publish(self._key("events", channel), json.dumps(event, default=str))
        except Exception as e:
            logger.warning(f"Redis publish error: {e}")

    def subscribe_events(self, channel: str, callback) -> None:
        with self._callbacks_lock:
            if channel not in self._callbacks:
                self._callbacks[channel] = []
            self._callbacks[channel].append(callback)

        # Start pub/sub thread if not running
        if self._pubsub is None:
            self._start_pubsub()

    def _start_pubsub(self) -> None:
        """Start the pub/sub listener thread."""
        try:
            pubsub = self._redis.pubsub()
            self._pubsub = pubsub

            # Subscribe to all registered channels
            with self._callbacks_lock:
                patterns = [self._key("events", "*")]

            pubsub.psubscribe(**{patterns[0]: self._handle_message})
            self._pubsub_thread = pubsub.run_in_thread(sleep_time=0.1)
            logger.info("Redis pub/sub listener started")
        except Exception as e:
            logger.warning(f"Failed to start Redis pub/sub: {e}")

    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming pub/sub message."""
        if message["type"] != "pmessage":
            return

        try:
            # Extract channel from pattern match
            full_channel = message["channel"]
            if isinstance(full_channel, bytes):
                full_channel = full_channel.decode()

            # Remove prefix to get logical channel
            prefix = self._key("events", "")
            channel = (
                full_channel[len(prefix) :] if full_channel.startswith(prefix) else full_channel
            )

            # Parse event data
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode()
            event = json.loads(data)

            # Call registered callbacks
            with self._callbacks_lock:
                callbacks = self._callbacks.get(channel, [])

            for cb in callbacks:
                try:
                    cb(event)
                except Exception as e:
                    logger.warning(f"Pub/sub callback error: {e}")
        except Exception as e:
            logger.warning(f"Failed to handle pub/sub message: {e}")

    def cleanup_expired(self) -> Dict[str, int]:
        """Redis handles expiry automatically via TTL."""
        return {"debate_states": 0, "active_loops": 0, "auth_states": 0}

    def close(self) -> None:
        """Close Redis connections."""
        if self._pubsub_thread:
            self._pubsub_thread.stop()
        if self._pubsub:
            self._pubsub.close()


# Global session store instance
_session_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_session_store(force_memory: bool = False) -> SessionStore:
    """Get the session store instance.

    Uses Redis if available and configured, otherwise falls back to in-memory.

    Args:
        force_memory: Force in-memory store even if Redis available

    Returns:
        SessionStore instance
    """
    global _session_store

    if _session_store is not None:
        return _session_store

    with _store_lock:
        if _session_store is not None:
            return _session_store

        if force_memory:
            _session_store = InMemorySessionStore()
            logger.info("Using in-memory session store (forced)")
            return _session_store

        # Try Redis first
        try:
            from aragora.server.redis_config import is_redis_available

            if is_redis_available():
                _session_store = RedisSessionStore()
                logger.info("Using Redis session store (distributed)")
                return _session_store
        except Exception as e:
            logger.debug(f"Redis session store unavailable: {e}")

        # Check if distributed state is required (multi-instance or production)
        if is_distributed_state_required():
            raise DistributedStateError(
                "session_store",
                "Redis not available for distributed session management",
            )

        # Fall back to in-memory (single instance only)
        logger.warning(
            "Using in-memory session store - sessions will be lost on restart "
            "and not shared across instances. Set ARAGORA_MULTI_INSTANCE=true "
            "and configure REDIS_URL for production deployments."
        )
        _session_store = InMemorySessionStore()
        return _session_store


def reset_session_store() -> None:
    """Reset session store (for testing)."""
    global _session_store
    with _store_lock:
        if _session_store is not None and hasattr(_session_store, "close"):
            _session_store.close()
        _session_store = None


__all__ = [
    "SessionStore",
    "SessionStoreConfig",
    "InMemorySessionStore",
    "RedisSessionStore",
    "get_session_store",
    "reset_session_store",
]
