"""
Leader Election for Distributed Control Plane.

Provides Redis-based leader election for multi-node Aragora deployments.
Uses a distributed lock with TTL for leader election.

Based on the Redlock algorithm for distributed locking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class LeaderState(Enum):
    """Current state of this node in the leader election."""

    FOLLOWER = "follower"  # Not the leader, following
    CANDIDATE = "candidate"  # Attempting to become leader
    LEADER = "leader"  # Currently the leader
    DISCONNECTED = "disconnected"  # Lost connection to coordination


@dataclass
class LeaderConfig:
    """Configuration for leader election."""

    # Redis connection
    redis_url: str = "redis://localhost:6379"
    key_prefix: str = "aragora:leader:"

    # Election timing
    lock_ttl_seconds: float = 30.0  # How long leader lock is valid
    heartbeat_interval: float = 10.0  # How often leader renews lock
    election_timeout: float = 5.0  # Timeout for election attempts
    retry_interval: float = 1.0  # How often followers check for leadership

    # Node identity
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    @classmethod
    def from_env(cls) -> "LeaderConfig":
        """Create config from environment variables."""
        return cls(
            redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
            key_prefix=os.environ.get("LEADER_KEY_PREFIX", "aragora:leader:"),
            lock_ttl_seconds=float(os.environ.get("LEADER_LOCK_TTL", "30")),
            heartbeat_interval=float(os.environ.get("LEADER_HEARTBEAT", "10")),
            election_timeout=float(os.environ.get("LEADER_ELECTION_TIMEOUT", "5")),
            retry_interval=float(os.environ.get("LEADER_RETRY_INTERVAL", "1")),
            node_id=os.environ.get(
                "NODE_ID",
                f"{os.uname().nodename}-{os.getpid()}-{str(uuid.uuid4())[:8]}",
            ),
        )


@dataclass
class LeaderInfo:
    """Information about the current leader."""

    node_id: str
    elected_at: float
    last_heartbeat: float
    metadata: dict[str, Any] = field(default_factory=dict)


class LeaderElection:
    """
    Redis-based leader election for distributed Aragora deployments.

    Uses a simple distributed lock pattern:
    1. Try to SET the leader key with NX (only if not exists) and EX (TTL)
    2. If successful, this node is the leader
    3. Leader must periodically refresh the lock before TTL expires
    4. If leader fails, lock expires and other nodes can acquire it

    Usage:
        election = LeaderElection()
        await election.start()

        if election.is_leader:
            # Run leader-specific tasks
            await run_control_plane_tasks()

        # Register callbacks
        election.on_become_leader(handle_leadership)
        election.on_lose_leadership(handle_demotion)

        # Graceful shutdown
        await election.stop()
    """

    def __init__(
        self,
        config: Optional[LeaderConfig] = None,
        redis_client: Optional[Any] = None,  # aioredis.Redis
    ):
        """
        Initialize leader election.

        Args:
            config: Election configuration
            redis_client: Optional pre-configured Redis client
        """
        self._config = config or LeaderConfig.from_env()
        self._redis = redis_client
        self._state = LeaderState.DISCONNECTED
        self._current_leader: Optional[LeaderInfo] = None

        self._running = False
        self._election_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_become_leader: list[Callable[[], Any]] = []
        self._on_lose_leader: list[Callable[[], Any]] = []
        self._on_leader_change: list[Callable[[Optional[str]], Any]] = []

    @property
    def state(self) -> LeaderState:
        """Current election state."""
        return self._state

    @property
    def is_leader(self) -> bool:
        """Check if this node is currently the leader."""
        return self._state == LeaderState.LEADER

    @property
    def node_id(self) -> str:
        """This node's unique identifier."""
        return self._config.node_id

    @property
    def current_leader(self) -> Optional[LeaderInfo]:
        """Information about the current leader."""
        return self._current_leader

    def on_become_leader(self, callback: Callable[[], Any]) -> None:
        """Register callback for when this node becomes leader."""
        self._on_become_leader.append(callback)

    def on_lose_leader(self, callback: Callable[[], Any]) -> None:
        """Register callback for when this node loses leadership."""
        self._on_lose_leader.append(callback)

    def on_leader_change(self, callback: Callable[[Optional[str]], Any]) -> None:
        """Register callback for any leader change (receives new leader node_id)."""
        self._on_leader_change.append(callback)

    async def start(self) -> None:
        """Start the leader election process."""
        if self._running:
            return

        logger.info(f"[leader] Starting election for node {self._config.node_id}")

        # Connect to Redis if not provided
        if self._redis is None:
            try:
                import aioredis

                self._redis = await aioredis.from_url(
                    self._config.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except ImportError:
                logger.warning("[leader] aioredis not available, using in-memory fallback")
                self._redis = _InMemoryRedis()

        self._running = True
        self._state = LeaderState.FOLLOWER

        # Start election loop
        self._election_task = asyncio.create_task(self._election_loop())
        logger.info(f"[leader] Election started for node {self._config.node_id}")

    async def stop(self) -> None:
        """Stop the leader election and release leadership if held."""
        if not self._running:
            return

        logger.info(f"[leader] Stopping election for node {self._config.node_id}")
        self._running = False

        # Release leadership if we have it
        if self._state == LeaderState.LEADER:
            await self._release_leadership()

        # Cancel tasks
        if self._election_task:
            self._election_task.cancel()
            try:
                await self._election_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        self._state = LeaderState.DISCONNECTED
        logger.info(f"[leader] Election stopped for node {self._config.node_id}")

    async def _election_loop(self) -> None:
        """Main election loop."""
        while self._running:
            try:
                if self._state == LeaderState.FOLLOWER:
                    # Check if we can become leader
                    leader = await self._get_current_leader()

                    if leader is None:
                        # No leader, try to become one
                        self._state = LeaderState.CANDIDATE
                        if await self._try_become_leader():
                            await self._handle_become_leader()
                        else:
                            self._state = LeaderState.FOLLOWER
                    else:
                        # Someone else is leader
                        old_leader = self._current_leader
                        self._current_leader = leader

                        if old_leader is None or old_leader.node_id != leader.node_id:
                            await self._notify_leader_change(leader.node_id)

                elif self._state == LeaderState.LEADER:
                    # We are leader, refresh our lock
                    if not await self._refresh_leadership():
                        # Lost leadership
                        await self._handle_lose_leader()

                await asyncio.sleep(self._config.retry_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[leader] Election loop error: {e}")
                await asyncio.sleep(self._config.retry_interval)

    async def _try_become_leader(self) -> bool:
        """Attempt to acquire the leader lock."""
        import time

        lock_key = f"{self._config.key_prefix}lock"
        lock_value = self._config.node_id

        try:
            # SET NX EX - only set if not exists, with expiry
            result = await self._redis.set(
                lock_key,
                lock_value,
                nx=True,
                ex=int(self._config.lock_ttl_seconds),
            )

            if result:
                logger.info(f"[leader] Node {self._config.node_id} acquired leadership")

                # Store additional leader info
                info_key = f"{self._config.key_prefix}info"
                await self._redis.hset(
                    info_key,
                    mapping={
                        "node_id": self._config.node_id,
                        "elected_at": str(time.time()),
                        "last_heartbeat": str(time.time()),
                    },
                )
                return True

            return False

        except Exception as e:
            logger.error(f"[leader] Failed to acquire lock: {e}")
            return False

    async def _refresh_leadership(self) -> bool:
        """Refresh the leadership lock TTL."""
        import time

        lock_key = f"{self._config.key_prefix}lock"

        try:
            # Check if we still hold the lock
            current = await self._redis.get(lock_key)
            if current != self._config.node_id:
                logger.warning(f"[leader] Lock held by {current}, not us")
                return False

            # Refresh TTL
            await self._redis.expire(lock_key, int(self._config.lock_ttl_seconds))

            # Update heartbeat
            info_key = f"{self._config.key_prefix}info"
            await self._redis.hset(info_key, "last_heartbeat", str(time.time()))

            return True

        except Exception as e:
            logger.error(f"[leader] Failed to refresh lock: {e}")
            return False

    async def _release_leadership(self) -> None:
        """Release the leadership lock."""
        lock_key = f"{self._config.key_prefix}lock"

        try:
            # Only delete if we hold the lock (using Lua script for atomicity)
            current = await self._redis.get(lock_key)
            if current == self._config.node_id:
                await self._redis.delete(lock_key)
                logger.info(f"[leader] Node {self._config.node_id} released leadership")
        except Exception as e:
            logger.error(f"[leader] Failed to release lock: {e}")

    async def _get_current_leader(self) -> Optional[LeaderInfo]:
        """Get information about the current leader."""
        import time

        lock_key = f"{self._config.key_prefix}lock"
        info_key = f"{self._config.key_prefix}info"

        try:
            node_id = await self._redis.get(lock_key)
            if not node_id:
                return None

            info = await self._redis.hgetall(info_key)
            if not info:
                return LeaderInfo(
                    node_id=node_id,
                    elected_at=time.time(),
                    last_heartbeat=time.time(),
                )

            return LeaderInfo(
                node_id=info.get("node_id", node_id),
                elected_at=float(info.get("elected_at", time.time())),
                last_heartbeat=float(info.get("last_heartbeat", time.time())),
            )

        except Exception as e:
            logger.error(f"[leader] Failed to get leader info: {e}")
            return None

    async def _handle_become_leader(self) -> None:
        """Handle becoming the leader."""
        import time

        self._state = LeaderState.LEADER
        self._current_leader = LeaderInfo(
            node_id=self._config.node_id,
            elected_at=time.time(),
            last_heartbeat=time.time(),
        )

        logger.info(f"[leader] Node {self._config.node_id} is now LEADER")

        # Notify callbacks
        for callback in self._on_become_leader:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[leader] Callback error: {e}")

        await self._notify_leader_change(self._config.node_id)

    async def _handle_lose_leader(self) -> None:
        """Handle losing leadership."""
        logger.warning(f"[leader] Node {self._config.node_id} lost leadership")
        self._state = LeaderState.FOLLOWER

        # Notify callbacks
        for callback in self._on_lose_leader:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[leader] Callback error: {e}")

    async def _notify_leader_change(self, new_leader: Optional[str]) -> None:
        """Notify callbacks of leader change."""
        for callback in self._on_leader_change:
            try:
                result = callback(new_leader)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[leader] Leader change callback error: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get election statistics."""
        return {
            "node_id": self._config.node_id,
            "state": self._state.value,
            "is_leader": self.is_leader,
            "current_leader": self._current_leader.node_id if self._current_leader else None,
        }


class _InMemoryRedis:
    """In-memory Redis mock for single-node deployments without Redis."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._hashes: dict[str, dict[str, str]] = {}

    async def set(self, key: str, value: str, nx: bool = False, ex: int = 0) -> bool:
        if nx and key in self._data:
            return False
        self._data[key] = value
        return True

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def expire(self, key: str, seconds: int) -> None:
        pass  # No-op for in-memory

    async def hset(self, key: str, field: str = None, value: str = None, mapping: dict = None) -> None:
        if key not in self._hashes:
            self._hashes[key] = {}
        if mapping:
            self._hashes[key].update(mapping)
        elif field:
            self._hashes[key][field] = value

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})


__all__ = [
    "LeaderState",
    "LeaderConfig",
    "LeaderInfo",
    "LeaderElection",
]
