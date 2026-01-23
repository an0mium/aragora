"""
Regional Synchronization for Multi-Region Control Plane.

Provides cross-region state synchronization with:
- Event-driven sync via Redis pub/sub
- Regional event bus for broadcasting changes
- Conflict-free state updates using timestamps
- Regional health monitoring

Usage:
    from aragora.control_plane.regional_sync import (
        RegionalEventBus,
        RegionalSyncConfig,
        RegionalStateManager,
    )

    # Create regional event bus
    config = RegionalSyncConfig(
        local_region="us-west-2",
        sync_regions=["us-east-1", "eu-west-1"],
    )
    event_bus = RegionalEventBus(redis_url="redis://localhost:6379", config=config)
    await event_bus.connect()

    # Publish state change
    await event_bus.publish_agent_update(agent_id, agent_state)

    # Subscribe to remote changes
    await event_bus.subscribe(handler_callback)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

# Observability
from aragora.observability import get_logger

logger = get_logger(__name__)


class RegionalEventType(Enum):
    """Types of regional synchronization events."""

    AGENT_REGISTERED = "agent_registered"
    AGENT_UPDATED = "agent_updated"
    AGENT_UNREGISTERED = "agent_unregistered"
    AGENT_HEARTBEAT = "agent_heartbeat"

    TASK_SUBMITTED = "task_submitted"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"

    LEADER_ELECTED = "leader_elected"
    LEADER_RESIGNED = "leader_resigned"

    REGION_HEALTH = "region_health"
    REGION_JOINED = "region_joined"
    REGION_LEFT = "region_left"


@dataclass
class RegionalEvent:
    """
    An event for cross-region synchronization.

    Uses timestamps for conflict-free ordering (last-write-wins).
    """

    event_type: RegionalEventType
    source_region: str
    entity_id: str  # Agent ID or Task ID
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    version: int = 1  # For future schema evolution

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.value,
            "source_region": self.source_region,
            "entity_id": self.entity_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionalEvent":
        """Deserialize from dictionary."""
        return cls(
            event_type=RegionalEventType(data["event_type"]),
            source_region=data["source_region"],
            entity_id=data["entity_id"],
            timestamp=data.get("timestamp", time.time()),
            data=data.get("data", {}),
            version=data.get("version", 1),
        )

    def is_newer_than(self, other: "RegionalEvent") -> bool:
        """Check if this event is newer than another (for conflict resolution)."""
        return self.timestamp > other.timestamp


@dataclass
class RegionalSyncConfig:
    """Configuration for regional synchronization."""

    # Local region identifier
    local_region: str = field(default_factory=lambda: os.environ.get("ARAGORA_REGION", "default"))

    # Regions to sync with (empty = all regions via broadcast)
    sync_regions: List[str] = field(default_factory=list)

    # Redis channel prefix for regional events
    channel_prefix: str = "aragora:regional:"

    # How often to publish heartbeats (seconds)
    heartbeat_interval: float = 10.0

    # How long before a region is considered unhealthy (seconds)
    region_timeout: float = 30.0

    # Maximum events to buffer locally when disconnected
    max_event_buffer: int = 1000

    # Whether to sync agent heartbeats (high volume)
    sync_heartbeats: bool = False

    # Conflict resolution strategy
    conflict_strategy: str = "last_write_wins"  # or "merge", "source_priority"

    def get_global_channel(self) -> str:
        """Get the global broadcast channel name."""
        return f"{self.channel_prefix}global"

    def get_region_channel(self, region_id: str) -> str:
        """Get region-specific channel name."""
        return f"{self.channel_prefix}region:{region_id}"


# Type alias for event handlers
RegionalEventHandler = Callable[[RegionalEvent], Coroutine[Any, Any, None]]


class RegionalEventBus:
    """
    Event bus for cross-region state synchronization.

    Uses Redis pub/sub for broadcasting events to all regions.
    Supports both global broadcast and region-specific channels.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        config: Optional[RegionalSyncConfig] = None,
    ):
        """
        Initialize the regional event bus.

        Args:
            redis_url: Redis connection URL
            config: Regional sync configuration
        """
        self._redis_url = redis_url
        self._config = config or RegionalSyncConfig()
        self._redis: Optional[Any] = None
        self._pubsub: Optional[Any] = None
        self._connected = False
        self._running = False

        # Event handlers
        self._handlers: Dict[RegionalEventType, List[RegionalEventHandler]] = {}
        self._global_handlers: List[RegionalEventHandler] = []

        # Event buffer for offline operation
        self._event_buffer: List[RegionalEvent] = []

        # Last seen timestamps per region (for health monitoring)
        self._region_last_seen: Dict[str, float] = {}

        # Background tasks
        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    @property
    def local_region(self) -> str:
        """Get the local region identifier."""
        return self._config.local_region

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    async def connect(self) -> bool:
        """
        Connect to Redis and start listening for events.

        Returns:
            True if connected successfully
        """
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            self._connected = True

            # Create pubsub connection
            self._pubsub = self._redis.pubsub()

            # Subscribe to global channel and local region channel
            await self._pubsub.subscribe(self._config.get_global_channel())
            await self._pubsub.subscribe(self._config.get_region_channel(self._config.local_region))

            logger.info(f"RegionalEventBus connected to Redis, region={self._config.local_region}")

            # Start background tasks
            self._running = True
            self._listener_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Announce region joined
            await self.publish(
                RegionalEvent(
                    event_type=RegionalEventType.REGION_JOINED,
                    source_region=self._config.local_region,
                    entity_id=self._config.local_region,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                )
            )

            # Flush buffered events
            await self._flush_buffer()

            return True

        except ImportError:
            logger.warning("redis package not installed, regional sync disabled")
            return False
        except Exception as e:
            logger.error(f"Failed to connect RegionalEventBus: {e}")
            return False

    async def close(self) -> None:
        """Close connections and stop background tasks."""
        self._running = False

        # Announce region leaving
        if self._connected:
            try:
                await self.publish(
                    RegionalEvent(
                        event_type=RegionalEventType.REGION_LEFT,
                        source_region=self._config.local_region,
                        entity_id=self._config.local_region,
                    )
                )
            except Exception:
                pass

        # Cancel background tasks
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close pubsub
        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()

        # Close Redis
        if self._redis:
            await self._redis.close()

        self._connected = False
        logger.info("RegionalEventBus disconnected")

    def subscribe(
        self,
        event_type: Optional[RegionalEventType] = None,
        handler: Optional[RegionalEventHandler] = None,
    ) -> None:
        """
        Subscribe to regional events.

        Args:
            event_type: Specific event type to subscribe to (None for all)
            handler: Async handler function
        """
        if handler is None:
            return

        if event_type is None:
            self._global_handlers.append(handler)
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: Optional[RegionalEventType] = None,
        handler: Optional[RegionalEventHandler] = None,
    ) -> None:
        """Unsubscribe a handler from events."""
        if handler is None:
            return

        if event_type is None:
            if handler in self._global_handlers:
                self._global_handlers.remove(handler)
        else:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    async def publish(
        self,
        event: RegionalEvent,
        target_region: Optional[str] = None,
    ) -> bool:
        """
        Publish an event to other regions.

        Args:
            event: Event to publish
            target_region: Specific region to send to (None = broadcast)

        Returns:
            True if published (or buffered)
        """
        # Don't sync heartbeats unless configured
        if (
            event.event_type == RegionalEventType.AGENT_HEARTBEAT
            and not self._config.sync_heartbeats
        ):
            return True

        if not self._connected:
            # Buffer event for later
            if len(self._event_buffer) < self._config.max_event_buffer:
                self._event_buffer.append(event)
                return True
            else:
                logger.warning("Event buffer full, dropping event")
                return False

        try:
            channel = (
                self._config.get_region_channel(target_region)
                if target_region
                else self._config.get_global_channel()
            )
            await self._redis.publish(channel, json.dumps(event.to_dict()))
            return True

        except Exception as e:
            logger.warning(f"Failed to publish regional event: {e}")
            # Buffer on failure
            if len(self._event_buffer) < self._config.max_event_buffer:
                self._event_buffer.append(event)
            return False

    async def publish_agent_update(
        self,
        agent_id: str,
        agent_data: Dict[str, Any],
        event_type: RegionalEventType = RegionalEventType.AGENT_UPDATED,
    ) -> bool:
        """Convenience method to publish agent state update."""
        event = RegionalEvent(
            event_type=event_type,
            source_region=self._config.local_region,
            entity_id=agent_id,
            data=agent_data,
        )
        return await self.publish(event)

    async def publish_task_update(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        event_type: RegionalEventType = RegionalEventType.TASK_SUBMITTED,
    ) -> bool:
        """Convenience method to publish task state update."""
        event = RegionalEvent(
            event_type=event_type,
            source_region=self._config.local_region,
            entity_id=task_id,
            data=task_data,
        )
        return await self.publish(event)

    def get_healthy_regions(self) -> List[str]:
        """Get list of regions that have sent heartbeats recently."""
        now = time.time()
        timeout = self._config.region_timeout
        return [
            region
            for region, last_seen in self._region_last_seen.items()
            if (now - last_seen) < timeout
        ]

    def get_region_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all known regions."""
        now = time.time()
        timeout = self._config.region_timeout
        result = {}

        for region, last_seen in self._region_last_seen.items():
            age = now - last_seen
            result[region] = {
                "last_seen": last_seen,
                "last_seen_ago_seconds": age,
                "healthy": age < timeout,
                "status": "healthy" if age < timeout else "unhealthy",
            }

        return result

    async def _listen_loop(self) -> None:
        """Background task to listen for events from other regions."""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message["type"] == "message":
                    await self._handle_message(message["data"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in regional event listener: {e}")
                await asyncio.sleep(1.0)

    async def _handle_message(self, data: str) -> None:
        """Handle an incoming message from Redis."""
        try:
            event_dict = json.loads(data)
            event = RegionalEvent.from_dict(event_dict)

            # Ignore events from self
            if event.source_region == self._config.local_region:
                return

            # Update region health tracking
            self._region_last_seen[event.source_region] = time.time()

            # Call handlers
            await self._dispatch_event(event)

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in regional event: {e}")
        except Exception as e:
            logger.warning(f"Error handling regional event: {e}")

    async def _dispatch_event(self, event: RegionalEvent) -> None:
        """Dispatch event to registered handlers."""
        # Call type-specific handlers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in regional event handler: {e}")

        # Call global handlers
        for handler in self._global_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error in global regional event handler: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background task to send periodic region heartbeats."""
        while self._running:
            try:
                await self.publish(
                    RegionalEvent(
                        event_type=RegionalEventType.REGION_HEALTH,
                        source_region=self._config.local_region,
                        entity_id=self._config.local_region,
                        data={
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "status": "healthy",
                        },
                    )
                )
                await asyncio.sleep(self._config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error sending region heartbeat: {e}")
                await asyncio.sleep(self._config.heartbeat_interval)

    async def _flush_buffer(self) -> None:
        """Flush buffered events after reconnection."""
        if not self._event_buffer:
            return

        logger.info(f"Flushing {len(self._event_buffer)} buffered regional events")
        events = self._event_buffer.copy()
        self._event_buffer.clear()

        for event in events:
            await self.publish(event)


@dataclass
class RegionHealth:
    """Health status of a region."""

    region_id: str
    last_seen: float
    healthy: bool
    agent_count: int = 0
    task_count: int = 0
    leader_id: Optional[str] = None


class RegionalStateManager:
    """
    Manages state synchronization across regions.

    Integrates with SharedControlPlaneState to keep agent and task
    state synchronized across multiple regional deployments.
    """

    def __init__(
        self,
        event_bus: RegionalEventBus,
        state_store: Optional[Any] = None,  # SharedControlPlaneState
    ):
        """
        Initialize the regional state manager.

        Args:
            event_bus: Regional event bus for cross-region communication
            state_store: Shared control plane state store
        """
        self._event_bus = event_bus
        self._state_store = state_store

        # Track entity versions for conflict detection
        self._entity_versions: Dict[str, float] = {}

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register handlers for regional events."""
        # Agent events
        self._event_bus.subscribe(RegionalEventType.AGENT_REGISTERED, self._handle_agent_registered)
        self._event_bus.subscribe(RegionalEventType.AGENT_UPDATED, self._handle_agent_updated)
        self._event_bus.subscribe(
            RegionalEventType.AGENT_UNREGISTERED, self._handle_agent_unregistered
        )

        # Task events
        self._event_bus.subscribe(RegionalEventType.TASK_SUBMITTED, self._handle_task_submitted)
        self._event_bus.subscribe(RegionalEventType.TASK_COMPLETED, self._handle_task_completed)

    async def _handle_agent_registered(self, event: RegionalEvent) -> None:
        """Handle agent registration from another region."""
        if self._state_store is None:
            return

        # Check if this is newer than our version
        if not self._is_newer(event.entity_id, event.timestamp):
            return

        self._entity_versions[event.entity_id] = event.timestamp
        logger.debug(f"Syncing agent {event.entity_id} from region {event.source_region}")

        # Update local state
        # Note: This would call state_store methods when integrated

    async def _handle_agent_updated(self, event: RegionalEvent) -> None:
        """Handle agent update from another region."""
        if not self._is_newer(event.entity_id, event.timestamp):
            return

        self._entity_versions[event.entity_id] = event.timestamp
        logger.debug(f"Syncing agent update {event.entity_id} from region {event.source_region}")

    async def _handle_agent_unregistered(self, event: RegionalEvent) -> None:
        """Handle agent unregistration from another region."""
        if not self._is_newer(event.entity_id, event.timestamp):
            return

        self._entity_versions[event.entity_id] = event.timestamp
        logger.debug(f"Syncing agent removal {event.entity_id} from region {event.source_region}")

    async def _handle_task_submitted(self, event: RegionalEvent) -> None:
        """Handle task submission from another region."""
        if not self._is_newer(event.entity_id, event.timestamp):
            return

        self._entity_versions[event.entity_id] = event.timestamp
        logger.debug(f"Syncing task {event.entity_id} from region {event.source_region}")

    async def _handle_task_completed(self, event: RegionalEvent) -> None:
        """Handle task completion from another region."""
        if not self._is_newer(event.entity_id, event.timestamp):
            return

        self._entity_versions[event.entity_id] = event.timestamp
        logger.debug(f"Syncing task completion {event.entity_id} from region {event.source_region}")

    def _is_newer(self, entity_id: str, timestamp: float) -> bool:
        """Check if an event timestamp is newer than our version."""
        current = self._entity_versions.get(entity_id, 0.0)
        return timestamp > current

    async def sync_task_state(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        event_type: RegionalEventType = RegionalEventType.TASK_SUBMITTED,
        target_region: Optional[str] = None,
    ) -> bool:
        """
        Synchronize task state to other regions.

        Args:
            task_id: Task identifier
            task_data: Task state data to sync
            event_type: Type of task event
            target_region: Specific region to sync to (None = broadcast)

        Returns:
            True if sync initiated successfully
        """
        # Update local version tracking
        self._entity_versions[task_id] = time.time()

        # Publish via event bus
        return await self._event_bus.publish_task_update(
            task_id=task_id,
            task_data=task_data,
            event_type=event_type,
        )

    async def sync_agent_state(
        self,
        agent_id: str,
        agent_data: Dict[str, Any],
        event_type: RegionalEventType = RegionalEventType.AGENT_UPDATED,
    ) -> bool:
        """
        Synchronize agent state to other regions.

        Args:
            agent_id: Agent identifier
            agent_data: Agent state data to sync
            event_type: Type of agent event

        Returns:
            True if sync initiated successfully
        """
        # Update local version tracking
        self._entity_versions[agent_id] = time.time()

        # Publish via event bus
        return await self._event_bus.publish_agent_update(
            agent_id=agent_id,
            agent_data=agent_data,
            event_type=event_type,
        )

    async def request_full_sync(self, target_region: str) -> bool:
        """
        Request full state sync from a specific region.

        Used during region startup or after network partition recovery.

        Args:
            target_region: Region to request sync from

        Returns:
            True if request sent successfully
        """
        event = RegionalEvent(
            event_type=RegionalEventType.REGION_JOINED,
            source_region=self._event_bus.local_region,
            entity_id=self._event_bus.local_region,
            data={
                "request_type": "full_sync",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return await self._event_bus.publish(event, target_region=target_region)

    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status and metrics."""
        return {
            "local_region": self._event_bus.local_region,
            "entities_tracked": len(self._entity_versions),
            "healthy_regions": self._event_bus.get_healthy_regions(),
            "region_health": self._event_bus.get_region_health(),
            "is_connected": self._event_bus.is_connected,
        }


# Module-level singleton
_regional_event_bus: Optional[RegionalEventBus] = None


def get_regional_event_bus() -> Optional[RegionalEventBus]:
    """Get the global regional event bus instance."""
    return _regional_event_bus


def set_regional_event_bus(bus: RegionalEventBus) -> None:
    """Set the global regional event bus instance."""
    global _regional_event_bus
    _regional_event_bus = bus


async def init_regional_sync(
    redis_url: str = "redis://localhost:6379",
    config: Optional[RegionalSyncConfig] = None,
) -> Optional[RegionalEventBus]:
    """
    Initialize regional synchronization.

    Args:
        redis_url: Redis connection URL
        config: Regional sync configuration

    Returns:
        RegionalEventBus if connected, None otherwise
    """
    bus = RegionalEventBus(redis_url=redis_url, config=config)
    if await bus.connect():
        set_regional_event_bus(bus)
        return bus
    return None
