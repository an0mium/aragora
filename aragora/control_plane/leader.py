"""
Leader Election for Distributed Control Plane.

Provides Redis-based leader election for multi-node Aragora deployments.
Uses a distributed lock with TTL for leader election.

Based on the Redlock algorithm for distributed locking.

SECURITY: In multi-instance mode (ARAGORA_MULTI_INSTANCE=true or production),
Redis is REQUIRED. Without it, each instance becomes its own leader, causing
split-brain scenarios. Set ARAGORA_SINGLE_INSTANCE=true for single-node deployments.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# Observability
from aragora.observability import get_logger

logger = get_logger(__name__)


def is_distributed_state_required() -> bool:
    """Check if distributed state backend (Redis) is required.

    Returns True if:
    - ARAGORA_REQUIRE_DISTRIBUTED_STATE=true
    - ARAGORA_MULTI_INSTANCE=true
    - ARAGORA_ENV=production (unless ARAGORA_SINGLE_INSTANCE=true)
    """
    if os.environ.get("ARAGORA_REQUIRE_DISTRIBUTED_STATE", "").lower() in ("true", "1", "yes"):
        return True
    if os.environ.get("ARAGORA_MULTI_INSTANCE", "").lower() in ("true", "1", "yes"):
        return True
    if os.environ.get("ARAGORA_ENV") == "production":
        if os.environ.get("ARAGORA_SINGLE_INSTANCE", "").lower() in ("true", "1", "yes"):
            return False
        return True
    return False


class DistributedStateError(Exception):
    """Raised when distributed state is required but not available."""

    def __init__(self, component: str, reason: str):
        self.component = component
        self.reason = reason
        super().__init__(
            f"Distributed state required for {component} but not available: {reason}. "
            f"Install aioredis and configure REDIS_URL, or set ARAGORA_SINGLE_INSTANCE=true."
        )


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
        """Start the leader election process.

        Raises:
            DistributedStateError: If Redis is required but not available.
        """
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
                logger.info("[leader] Connected to Redis for distributed leader election")
            except ImportError:
                if is_distributed_state_required():
                    raise DistributedStateError(
                        "leader_election",
                        "aioredis not installed. Install with: pip install aioredis",
                    )
                logger.warning(
                    "[leader] aioredis not available, using in-memory fallback. "
                    "This is NOT suitable for multi-instance deployments! "
                    "Set ARAGORA_SINGLE_INSTANCE=true to suppress this warning."
                )
                self._redis = _InMemoryRedis()
            except Exception as e:
                if is_distributed_state_required():
                    raise DistributedStateError(
                        "leader_election",
                        f"Failed to connect to Redis: {e}",
                    ) from e
                logger.warning(f"[leader] Redis connection failed, using in-memory fallback: {e}")
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

    async def hset(
        self, key: str, field: str = None, value: str = None, mapping: dict = None
    ) -> None:
        if key not in self._hashes:
            self._hashes[key] = {}
        if mapping:
            self._hashes[key].update(mapping)
        elif field:
            self._hashes[key][field] = value

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})


@dataclass
class RegionalLeaderConfig(LeaderConfig):
    """Configuration for regional leader election."""

    region_id: str = "default"
    sync_regions: list[str] = field(default_factory=list)
    broadcast_leadership: bool = True  # Broadcast leadership changes via event bus

    def get_region_key_prefix(self) -> str:
        """Get region-scoped key prefix."""
        return f"{self.key_prefix}region:{self.region_id}:"

    @classmethod
    def from_env(cls) -> "RegionalLeaderConfig":
        """Create config from environment variables."""
        base = LeaderConfig.from_env()
        return cls(
            redis_url=base.redis_url,
            key_prefix=base.key_prefix,
            lock_ttl_seconds=base.lock_ttl_seconds,
            heartbeat_interval=base.heartbeat_interval,
            election_timeout=base.election_timeout,
            retry_interval=base.retry_interval,
            node_id=base.node_id,
            region_id=os.environ.get("ARAGORA_REGION_ID", "default"),
            sync_regions=[
                r.strip()
                for r in os.environ.get("ARAGORA_SYNC_REGIONS", "").split(",")
                if r.strip()
            ],
            broadcast_leadership=os.environ.get("ARAGORA_BROADCAST_LEADERSHIP", "true").lower()
            in ("true", "1", "yes"),
        )


@dataclass
class RegionalLeaderInfo(LeaderInfo):
    """Information about a regional leader."""

    region_id: str = "default"
    is_global_coordinator: bool = False  # If this region is the global coordinator


class RegionalLeaderElection(LeaderElection):
    """
    Per-region leader election for multi-region deployments.

    Each region has its own leader, enabling:
    - Regional autonomy for local operations
    - Reduced cross-region latency
    - Fault isolation per region

    Additionally, a global coordinator can be elected from regional leaders
    for cross-region operations.

    Usage:
        election = RegionalLeaderElection(
            config=RegionalLeaderConfig(region_id="us-west-2")
        )
        await election.start()

        if election.is_regional_leader:
            # Handle regional leadership tasks
            pass

        if election.is_global_coordinator:
            # Handle global coordination tasks
            pass
    """

    def __init__(
        self,
        config: Optional[RegionalLeaderConfig] = None,
        redis_client: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        Initialize regional leader election.

        Args:
            config: Regional election configuration
            redis_client: Optional pre-configured Redis client
            event_bus: Optional RegionalEventBus for broadcasting
        """
        self._regional_config = config or RegionalLeaderConfig.from_env()
        super().__init__(
            config=self._regional_config,
            redis_client=redis_client,
        )

        self._event_bus = event_bus
        self._is_global_coordinator = False
        self._regional_leaders: dict[str, RegionalLeaderInfo] = {}

        # Callbacks for regional events
        self._on_become_regional_leader: list[Callable[[], Any]] = []
        self._on_lose_regional_leader: list[Callable[[], Any]] = []
        self._on_become_global_coordinator: list[Callable[[], Any]] = []
        self._on_lose_global_coordinator: list[Callable[[], Any]] = []

    @property
    def region_id(self) -> str:
        """This node's region identifier."""
        return self._regional_config.region_id

    @property
    def is_regional_leader(self) -> bool:
        """Check if this node is the leader of its region."""
        return self.is_leader

    @property
    def is_global_coordinator(self) -> bool:
        """Check if this node is the global coordinator."""
        return self._is_global_coordinator

    @property
    def regional_leaders(self) -> dict[str, RegionalLeaderInfo]:
        """Get info about all known regional leaders."""
        return self._regional_leaders.copy()

    def on_become_regional_leader(self, callback: Callable[[], Any]) -> None:
        """Register callback for when this node becomes regional leader."""
        self._on_become_regional_leader.append(callback)

    def on_lose_regional_leader(self, callback: Callable[[], Any]) -> None:
        """Register callback for when this node loses regional leadership."""
        self._on_lose_regional_leader.append(callback)

    def on_become_global_coordinator(self, callback: Callable[[], Any]) -> None:
        """Register callback for when this node becomes global coordinator."""
        self._on_become_global_coordinator.append(callback)

    def on_lose_global_coordinator(self, callback: Callable[[], Any]) -> None:
        """Register callback for when this node loses global coordinator role."""
        self._on_lose_global_coordinator.append(callback)

    async def start(self) -> None:
        """Start the regional leader election process."""
        # Override key prefix to be region-scoped
        original_prefix = self._config.key_prefix
        self._config.key_prefix = self._regional_config.get_region_key_prefix()

        logger.info(
            f"[regional-leader] Starting election for node {self._config.node_id} "
            f"in region {self.region_id}"
        )

        await super().start()

        # Restore original prefix for global coordinator election
        self._config.key_prefix = original_prefix

        # Register for event bus updates if available
        await self._subscribe_to_leadership_events()

    async def _subscribe_to_leadership_events(self) -> None:
        """Subscribe to leadership events from other regions."""
        if self._event_bus is None:
            return

        try:
            from aragora.control_plane.regional_sync import RegionalEventType

            # Subscribe to leader elected events
            self._event_bus.subscribe(
                RegionalEventType.LEADER_ELECTED,
                self._handle_remote_leader_elected,
            )
            self._event_bus.subscribe(
                RegionalEventType.LEADER_RESIGNED,
                self._handle_remote_leader_resigned,
            )
            logger.debug("[regional-leader] Subscribed to leadership events")
        except ImportError:
            logger.debug("[regional-leader] RegionalEventBus not available")

    async def _handle_remote_leader_elected(self, event: Any) -> None:
        """Handle leader election from another region."""
        import time

        source_region = event.source_region
        if source_region == self.region_id:
            return  # Ignore our own region

        self._regional_leaders[source_region] = RegionalLeaderInfo(
            node_id=event.entity_id,
            region_id=source_region,
            elected_at=event.timestamp,
            last_heartbeat=time.time(),
            is_global_coordinator=event.data.get("is_global_coordinator", False),
        )
        logger.debug(f"[regional-leader] Region {source_region} elected leader: {event.entity_id}")

    async def _handle_remote_leader_resigned(self, event: Any) -> None:
        """Handle leader resignation from another region."""
        source_region = event.source_region
        if source_region in self._regional_leaders:
            del self._regional_leaders[source_region]
            logger.debug(f"[regional-leader] Region {source_region} leader resigned")

    async def _handle_become_leader(self) -> None:
        """Handle becoming the regional leader."""
        await super()._handle_become_leader()

        # Notify regional callbacks
        for callback in self._on_become_regional_leader:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[regional-leader] Regional callback error: {e}")

        # Broadcast leadership via event bus
        await self._broadcast_leadership_change(elected=True)

        # Try to become global coordinator if we're regional leader
        await self._try_become_global_coordinator()

    async def _handle_lose_leader(self) -> None:
        """Handle losing regional leadership."""
        was_global = self._is_global_coordinator

        # Lose global coordinator role if we had it
        if self._is_global_coordinator:
            await self._release_global_coordinator()

        await super()._handle_lose_leader()

        # Notify regional callbacks
        for callback in self._on_lose_regional_leader:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[regional-leader] Regional callback error: {e}")

        # Broadcast leadership change
        await self._broadcast_leadership_change(elected=False)

        if was_global:
            for callback in self._on_lose_global_coordinator:
                try:
                    result = callback()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"[regional-leader] Global coordinator callback error: {e}")

    async def _broadcast_leadership_change(self, elected: bool) -> None:
        """Broadcast leadership change to other regions."""
        if self._event_bus is None or not self._regional_config.broadcast_leadership:
            return

        try:
            from aragora.control_plane.regional_sync import (
                RegionalEvent,
                RegionalEventType,
            )

            event_type = (
                RegionalEventType.LEADER_ELECTED if elected else RegionalEventType.LEADER_RESIGNED
            )
            event = RegionalEvent(
                event_type=event_type,
                source_region=self.region_id,
                entity_id=self._config.node_id,
                data={
                    "is_global_coordinator": self._is_global_coordinator,
                    "region_id": self.region_id,
                },
            )
            await self._event_bus.publish(event)
            logger.debug(
                f"[regional-leader] Broadcast leadership {'elected' if elected else 'resigned'}"
            )
        except Exception as e:
            logger.warning(f"[regional-leader] Failed to broadcast leadership: {e}")

    async def _try_become_global_coordinator(self) -> None:
        """Try to become the global coordinator (elected from regional leaders)."""
        if not self.is_regional_leader:
            return

        import time

        global_key = f"{self._config.key_prefix}global:coordinator"

        try:
            # Try to acquire global coordinator lock
            result = await self._redis.set(
                global_key,
                self._config.node_id,
                nx=True,
                ex=int(self._regional_config.lock_ttl_seconds * 2),  # Longer TTL for global
            )

            if result:
                self._is_global_coordinator = True
                logger.info(
                    f"[regional-leader] Node {self._config.node_id} is now GLOBAL COORDINATOR"
                )

                # Store coordinator info
                info_key = f"{self._config.key_prefix}global:coordinator_info"
                await self._redis.hset(
                    info_key,
                    mapping={
                        "node_id": self._config.node_id,
                        "region_id": self.region_id,
                        "elected_at": str(time.time()),
                    },
                )

                # Notify callbacks
                for callback in self._on_become_global_coordinator:
                    try:
                        result = callback()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"[regional-leader] Global coordinator callback error: {e}")

                # Broadcast updated status
                await self._broadcast_leadership_change(elected=True)

        except Exception as e:
            logger.debug(f"[regional-leader] Failed to become global coordinator: {e}")

    async def _release_global_coordinator(self) -> None:
        """Release the global coordinator role."""
        if not self._is_global_coordinator:
            return

        global_key = f"{self._config.key_prefix}global:coordinator"

        try:
            current = await self._redis.get(global_key)
            if current == self._config.node_id:
                await self._redis.delete(global_key)
                logger.info(
                    f"[regional-leader] Node {self._config.node_id} released global coordinator"
                )
        except Exception as e:
            logger.error(f"[regional-leader] Failed to release global coordinator: {e}")

        self._is_global_coordinator = False

    async def get_global_coordinator(self) -> Optional[RegionalLeaderInfo]:
        """Get information about the current global coordinator."""
        import time

        global_key = f"{self._config.key_prefix}global:coordinator"
        info_key = f"{self._config.key_prefix}global:coordinator_info"

        try:
            node_id = await self._redis.get(global_key)
            if not node_id:
                return None

            info = await self._redis.hgetall(info_key)
            return RegionalLeaderInfo(
                node_id=info.get("node_id", node_id),
                region_id=info.get("region_id", "unknown"),
                elected_at=float(info.get("elected_at", time.time())),
                last_heartbeat=time.time(),
                is_global_coordinator=True,
            )
        except Exception as e:
            logger.error(f"[regional-leader] Failed to get global coordinator: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get regional election statistics."""
        base_stats = super().get_stats()
        base_stats.update(
            {
                "region_id": self.region_id,
                "is_regional_leader": self.is_regional_leader,
                "is_global_coordinator": self.is_global_coordinator,
                "known_regional_leaders": list(self._regional_leaders.keys()),
            }
        )
        return base_stats


# Singleton for regional leader election
_regional_leader_election: Optional[RegionalLeaderElection] = None


def get_regional_leader_election() -> Optional[RegionalLeaderElection]:
    """Get the global regional leader election instance."""
    return _regional_leader_election


def set_regional_leader_election(election: RegionalLeaderElection) -> None:
    """Set the global regional leader election instance."""
    global _regional_leader_election
    _regional_leader_election = election


async def init_regional_leader_election(
    region_id: Optional[str] = None,
    event_bus: Optional[Any] = None,
) -> Optional[RegionalLeaderElection]:
    """
    Initialize and start regional leader election.

    Args:
        region_id: Optional region ID (defaults to ARAGORA_REGION_ID env var)
        event_bus: Optional RegionalEventBus for cross-region communication

    Returns:
        RegionalLeaderElection instance if started, None otherwise
    """
    global _regional_leader_election

    if _regional_leader_election is not None:
        return _regional_leader_election

    config = RegionalLeaderConfig.from_env()
    if region_id:
        config.region_id = region_id

    election = RegionalLeaderElection(config=config, event_bus=event_bus)

    try:
        await election.start()
        _regional_leader_election = election
        logger.info(f"[regional-leader] Initialized election for region {config.region_id}")
        return election
    except Exception as e:
        logger.warning(f"[regional-leader] Failed to initialize: {e}")
        return None


__all__ = [
    "LeaderState",
    "LeaderConfig",
    "LeaderInfo",
    "LeaderElection",
    "DistributedStateError",
    "is_distributed_state_required",
    # Regional
    "RegionalLeaderConfig",
    "RegionalLeaderInfo",
    "RegionalLeaderElection",
    "get_regional_leader_election",
    "set_regional_leader_election",
    "init_regional_leader_election",
]
