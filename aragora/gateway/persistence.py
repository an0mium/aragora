"""
Gateway Persistence Layer - Session and message durability.

Pattern: Local-first Persistence
Inspired by: Moltbot (https://github.com/moltbot)
Aragora adaptation: Multi-backend storage with local-first default

Provides persistent storage for gateway state:
- InMemoryGatewayStore: Fast ephemeral storage (testing)
- FileGatewayStore: JSON-based local file storage (local-first)
- RedisGatewayStore: Distributed cache with TTL (production)

Persisted state includes:
- Inbox messages (with TTL-based expiration)
- Device registry (paired devices)
- Routing rules (agent routing configuration)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from aragora.gateway.inbox import InboxMessage, MessagePriority
from aragora.gateway.device_registry import DeviceNode, DeviceStatus
from aragora.gateway.router import RoutingRule

logger = logging.getLogger(__name__)


# =============================================================================
# Store Protocol
# =============================================================================


@runtime_checkable
class GatewayStore(Protocol):
    """Protocol for gateway persistence backends."""

    async def save_message(self, message: InboxMessage) -> None:
        """Save an inbox message."""
        ...

    async def load_messages(self, limit: int = 1000) -> list[InboxMessage]:
        """Load inbox messages."""
        ...

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message by ID."""
        ...

    async def clear_messages(self, older_than_seconds: float | None = None) -> int:
        """Clear messages, optionally older than a threshold."""
        ...

    async def save_device(self, device: DeviceNode) -> None:
        """Save a device registration."""
        ...

    async def load_devices(self) -> list[DeviceNode]:
        """Load all registered devices."""
        ...

    async def delete_device(self, device_id: str) -> bool:
        """Delete a device registration."""
        ...

    async def save_rule(self, rule: RoutingRule) -> None:
        """Save a routing rule."""
        ...

    async def load_rules(self) -> list[RoutingRule]:
        """Load all routing rules."""
        ...

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a routing rule."""
        ...

    async def close(self) -> None:
        """Close the store and release resources."""
        ...


# =============================================================================
# Serialization Helpers
# =============================================================================


def _message_to_dict(msg: InboxMessage) -> dict[str, Any]:
    """Convert InboxMessage to serializable dict."""
    return {
        "message_id": msg.message_id,
        "channel": msg.channel,
        "sender": msg.sender,
        "content": msg.content,
        "timestamp": msg.timestamp,
        "is_read": msg.is_read,
        "is_replied": msg.is_replied,
        "priority": msg.priority.value
        if isinstance(msg.priority, MessagePriority)
        else msg.priority,
        "thread_id": msg.thread_id,
        "metadata": msg.metadata,
    }


def _dict_to_message(d: dict[str, Any]) -> InboxMessage:
    """Convert dict to InboxMessage."""
    priority = d.get("priority", 2)
    if isinstance(priority, int):
        priority = MessagePriority(priority)
    elif isinstance(priority, str):
        priority = MessagePriority[priority.upper()]

    return InboxMessage(
        message_id=d["message_id"],
        channel=d["channel"],
        sender=d["sender"],
        content=d["content"],
        timestamp=d.get("timestamp", time.time()),
        is_read=d.get("is_read", False),
        is_replied=d.get("is_replied", False),
        priority=priority,
        thread_id=d.get("thread_id"),
        metadata=d.get("metadata", {}),
    )


def _device_to_dict(device: DeviceNode) -> dict[str, Any]:
    """Convert DeviceNode to serializable dict."""
    return {
        "device_id": device.device_id,
        "name": device.name,
        "device_type": device.device_type,
        "status": device.status.value if isinstance(device.status, DeviceStatus) else device.status,
        "capabilities": device.capabilities,
        "last_seen": device.last_seen,
        "metadata": device.metadata,
    }


def _dict_to_device(d: dict[str, Any]) -> DeviceNode:
    """Convert dict to DeviceNode."""
    status = d.get("status", "offline")
    if isinstance(status, str):
        status = DeviceStatus(status)

    return DeviceNode(
        device_id=d.get("device_id"),
        name=d["name"],
        device_type=d.get("device_type", "unknown"),
        status=status,
        capabilities=d.get("capabilities", []),
        last_seen=d.get("last_seen"),
        metadata=d.get("metadata", {}),
    )


def _rule_to_dict(rule: RoutingRule) -> dict[str, Any]:
    """Convert RoutingRule to serializable dict."""
    return {
        "rule_id": rule.rule_id,
        "agent_id": rule.agent_id,
        "channel_pattern": rule.channel_pattern,
        "sender_pattern": rule.sender_pattern,
        "content_pattern": rule.content_pattern,
        "priority": rule.priority,
        "enabled": rule.enabled,
    }


def _dict_to_rule(d: dict[str, Any]) -> RoutingRule:
    """Convert dict to RoutingRule."""
    return RoutingRule(
        rule_id=d["rule_id"],
        agent_id=d["agent_id"],
        channel_pattern=d.get("channel_pattern"),
        sender_pattern=d.get("sender_pattern"),
        content_pattern=d.get("content_pattern"),
        priority=d.get("priority", 0),
        enabled=d.get("enabled", True),
    )


# =============================================================================
# In-Memory Store (Testing)
# =============================================================================


class InMemoryGatewayStore:
    """
    In-memory gateway store for testing.

    Fast but non-persistent - data is lost on restart.
    """

    def __init__(self) -> None:
        self._messages: dict[str, InboxMessage] = {}
        self._devices: dict[str, DeviceNode] = {}
        self._rules: dict[str, RoutingRule] = {}
        self._lock = asyncio.Lock()

    async def save_message(self, message: InboxMessage) -> None:
        """Save an inbox message."""
        async with self._lock:
            self._messages[message.message_id] = message

    async def load_messages(self, limit: int = 1000) -> list[InboxMessage]:
        """Load inbox messages."""
        async with self._lock:
            messages = list(self._messages.values())
            # Sort by timestamp descending (newest first)
            messages.sort(key=lambda m: m.timestamp, reverse=True)
            return messages[:limit]

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message by ID."""
        async with self._lock:
            if message_id in self._messages:
                del self._messages[message_id]
                return True
            return False

    async def clear_messages(self, older_than_seconds: float | None = None) -> int:
        """Clear messages, optionally older than a threshold."""
        async with self._lock:
            if older_than_seconds is None:
                count = len(self._messages)
                self._messages.clear()
                return count

            cutoff = time.time() - older_than_seconds
            to_delete = [mid for mid, msg in self._messages.items() if msg.timestamp < cutoff]
            for mid in to_delete:
                del self._messages[mid]
            return len(to_delete)

    async def save_device(self, device: DeviceNode) -> None:
        """Save a device registration."""
        async with self._lock:
            self._devices[device.device_id] = device

    async def load_devices(self) -> list[DeviceNode]:
        """Load all registered devices."""
        async with self._lock:
            return list(self._devices.values())

    async def delete_device(self, device_id: str) -> bool:
        """Delete a device registration."""
        async with self._lock:
            if device_id in self._devices:
                del self._devices[device_id]
                return True
            return False

    async def save_rule(self, rule: RoutingRule) -> None:
        """Save a routing rule."""
        async with self._lock:
            self._rules[rule.rule_id] = rule

    async def load_rules(self) -> list[RoutingRule]:
        """Load all routing rules."""
        async with self._lock:
            rules = list(self._rules.values())
            rules.sort(key=lambda r: r.priority, reverse=True)
            return rules

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a routing rule."""
        async with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False

    async def close(self) -> None:
        """Close the store (no-op for in-memory)."""
        pass


# =============================================================================
# File Store (Local-First)
# =============================================================================


class FileGatewayStore:
    """
    JSON file-based gateway store for local-first persistence.

    Stores gateway state in a local JSON file that survives restarts.
    Suitable for single-device deployments and development.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        auto_save: bool = True,
        auto_save_interval: float = 5.0,
    ) -> None:
        """
        Initialize file-based gateway store.

        Args:
            path: Path to the JSON storage file. Defaults to ~/.aragora/gateway.json
            auto_save: Whether to auto-save on writes.
            auto_save_interval: Minimum seconds between auto-saves.
        """
        if path is None:
            path = Path.home() / ".aragora" / "gateway.json"
        self._path = Path(path)
        self._auto_save = auto_save
        self._auto_save_interval = auto_save_interval

        self._messages: dict[str, InboxMessage] = {}
        self._devices: dict[str, DeviceNode] = {}
        self._rules: dict[str, RoutingRule] = {}

        self._lock = asyncio.Lock()
        self._dirty = False
        self._last_save = 0.0
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Ensure data is loaded from disk."""
        if self._loaded:
            return
        await self._load()
        self._loaded = True

    async def _load(self) -> None:
        """Load state from disk."""
        if not self._path.exists():
            return

        try:
            with open(self._path, "r") as f:
                data = json.load(f)

            # Load messages
            for msg_data in data.get("messages", []):
                try:
                    msg = _dict_to_message(msg_data)
                    self._messages[msg.message_id] = msg
                except Exception as e:
                    logger.warning(f"Failed to load message: {e}")

            # Load devices
            for dev_data in data.get("devices", []):
                try:
                    device = _dict_to_device(dev_data)
                    self._devices[device.device_id] = device
                except Exception as e:
                    logger.warning(f"Failed to load device: {e}")

            # Load rules
            for rule_data in data.get("rules", []):
                try:
                    rule = _dict_to_rule(rule_data)
                    self._rules[rule.rule_id] = rule
                except Exception as e:
                    logger.warning(f"Failed to load rule: {e}")

            logger.debug(
                f"Loaded gateway state: {len(self._messages)} messages, "
                f"{len(self._devices)} devices, {len(self._rules)} rules"
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse gateway state file: {e}")
        except Exception as e:
            logger.error(f"Failed to load gateway state: {e}")

    async def _save(self, force: bool = False) -> None:
        """Save state to disk."""
        if not force and not self._dirty:
            return

        now = time.time()
        if not force and (now - self._last_save) < self._auto_save_interval:
            return

        try:
            # Ensure directory exists
            self._path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": 1,
                "saved_at": now,
                "messages": [_message_to_dict(m) for m in self._messages.values()],
                "devices": [_device_to_dict(d) for d in self._devices.values()],
                "rules": [_rule_to_dict(r) for r in self._rules.values()],
            }

            # Write atomically via temp file
            temp_path = self._path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.rename(self._path)

            self._dirty = False
            self._last_save = now
            logger.debug(f"Saved gateway state to {self._path}")
        except Exception as e:
            logger.error(f"Failed to save gateway state: {e}")

    async def save_message(self, message: InboxMessage) -> None:
        """Save an inbox message."""
        async with self._lock:
            await self._ensure_loaded()
            self._messages[message.message_id] = message
            self._dirty = True
            if self._auto_save:
                await self._save()

    async def load_messages(self, limit: int = 1000) -> list[InboxMessage]:
        """Load inbox messages."""
        async with self._lock:
            await self._ensure_loaded()
            messages = list(self._messages.values())
            messages.sort(key=lambda m: m.timestamp, reverse=True)
            return messages[:limit]

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message by ID."""
        async with self._lock:
            await self._ensure_loaded()
            if message_id in self._messages:
                del self._messages[message_id]
                self._dirty = True
                if self._auto_save:
                    await self._save()
                return True
            return False

    async def clear_messages(self, older_than_seconds: float | None = None) -> int:
        """Clear messages, optionally older than a threshold."""
        async with self._lock:
            await self._ensure_loaded()
            if older_than_seconds is None:
                count = len(self._messages)
                self._messages.clear()
            else:
                cutoff = time.time() - older_than_seconds
                to_delete = [mid for mid, msg in self._messages.items() if msg.timestamp < cutoff]
                for mid in to_delete:
                    del self._messages[mid]
                count = len(to_delete)

            if count > 0:
                self._dirty = True
                if self._auto_save:
                    await self._save()
            return count

    async def save_device(self, device: DeviceNode) -> None:
        """Save a device registration."""
        async with self._lock:
            await self._ensure_loaded()
            self._devices[device.device_id] = device
            self._dirty = True
            if self._auto_save:
                await self._save()

    async def load_devices(self) -> list[DeviceNode]:
        """Load all registered devices."""
        async with self._lock:
            await self._ensure_loaded()
            return list(self._devices.values())

    async def delete_device(self, device_id: str) -> bool:
        """Delete a device registration."""
        async with self._lock:
            await self._ensure_loaded()
            if device_id in self._devices:
                del self._devices[device_id]
                self._dirty = True
                if self._auto_save:
                    await self._save()
                return True
            return False

    async def save_rule(self, rule: RoutingRule) -> None:
        """Save a routing rule."""
        async with self._lock:
            await self._ensure_loaded()
            self._rules[rule.rule_id] = rule
            self._dirty = True
            if self._auto_save:
                await self._save()

    async def load_rules(self) -> list[RoutingRule]:
        """Load all routing rules."""
        async with self._lock:
            await self._ensure_loaded()
            rules = list(self._rules.values())
            rules.sort(key=lambda r: r.priority, reverse=True)
            return rules

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a routing rule."""
        async with self._lock:
            await self._ensure_loaded()
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._dirty = True
                if self._auto_save:
                    await self._save()
                return True
            return False

    async def close(self) -> None:
        """Close the store and save any pending changes."""
        async with self._lock:
            if self._dirty:
                await self._save(force=True)


# =============================================================================
# Redis Store (Production)
# =============================================================================


class RedisGatewayStore:
    """
    Redis-based gateway store for production deployments.

    Provides distributed state with TTL-based expiration.
    Requires redis-py async support.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "aragora:gateway:",
        message_ttl_seconds: int = 86400 * 7,  # 7 days
        device_ttl_seconds: int = 86400 * 30,  # 30 days
    ) -> None:
        """
        Initialize Redis gateway store.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for all Redis keys.
            message_ttl_seconds: TTL for message keys.
            device_ttl_seconds: TTL for device keys.
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._message_ttl = message_ttl_seconds
        self._device_ttl = device_ttl_seconds
        self._redis: Any = None

    async def _get_redis(self) -> Any:
        """Get or create Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(self._redis_url)
            except ImportError:
                raise ImportError(
                    "redis-py with async support required. Install with: pip install redis[async]"
                )
        return self._redis

    def _msg_key(self, message_id: str) -> str:
        return f"{self._key_prefix}msg:{message_id}"

    def _msg_index_key(self) -> str:
        return f"{self._key_prefix}msg:index"

    def _dev_key(self, device_id: str) -> str:
        return f"{self._key_prefix}dev:{device_id}"

    def _dev_index_key(self) -> str:
        return f"{self._key_prefix}dev:index"

    def _rule_key(self, rule_id: str) -> str:
        return f"{self._key_prefix}rule:{rule_id}"

    def _rule_index_key(self) -> str:
        return f"{self._key_prefix}rule:index"

    async def save_message(self, message: InboxMessage) -> None:
        """Save an inbox message."""
        redis = await self._get_redis()
        key = self._msg_key(message.message_id)
        data = json.dumps(_message_to_dict(message))

        pipe = redis.pipeline()
        pipe.set(key, data, ex=self._message_ttl)
        pipe.zadd(self._msg_index_key(), {message.message_id: message.timestamp})
        await pipe.execute()

    async def load_messages(self, limit: int = 1000) -> list[InboxMessage]:
        """Load inbox messages."""
        redis = await self._get_redis()

        # Get message IDs from sorted set (newest first)
        msg_ids = await redis.zrevrange(self._msg_index_key(), 0, limit - 1)
        if not msg_ids:
            return []

        # Get message data
        keys = [self._msg_key(mid.decode() if isinstance(mid, bytes) else mid) for mid in msg_ids]
        data_list = await redis.mget(keys)

        messages = []
        for data in data_list:
            if data:
                try:
                    msg = _dict_to_message(json.loads(data))
                    messages.append(msg)
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}")
        return messages

    async def delete_message(self, message_id: str) -> bool:
        """Delete a message by ID."""
        redis = await self._get_redis()
        key = self._msg_key(message_id)

        pipe = redis.pipeline()
        pipe.delete(key)
        pipe.zrem(self._msg_index_key(), message_id)
        results = await pipe.execute()
        return results[0] > 0

    async def clear_messages(self, older_than_seconds: float | None = None) -> int:
        """Clear messages, optionally older than a threshold."""
        redis = await self._get_redis()

        if older_than_seconds is None:
            # Get all message IDs
            msg_ids = await redis.zrange(self._msg_index_key(), 0, -1)
            if not msg_ids:
                return 0

            keys = [
                self._msg_key(mid.decode() if isinstance(mid, bytes) else mid) for mid in msg_ids
            ]
            pipe = redis.pipeline()
            pipe.delete(*keys)
            pipe.delete(self._msg_index_key())
            await pipe.execute()
            return len(msg_ids)
        else:
            cutoff = time.time() - older_than_seconds
            # Get IDs older than cutoff
            msg_ids = await redis.zrangebyscore(self._msg_index_key(), "-inf", cutoff)
            if not msg_ids:
                return 0

            keys = [
                self._msg_key(mid.decode() if isinstance(mid, bytes) else mid) for mid in msg_ids
            ]
            pipe = redis.pipeline()
            pipe.delete(*keys)
            pipe.zremrangebyscore(self._msg_index_key(), "-inf", cutoff)
            await pipe.execute()
            return len(msg_ids)

    async def save_device(self, device: DeviceNode) -> None:
        """Save a device registration."""
        redis = await self._get_redis()
        key = self._dev_key(device.device_id)
        data = json.dumps(_device_to_dict(device))

        pipe = redis.pipeline()
        pipe.set(key, data, ex=self._device_ttl)
        pipe.sadd(self._dev_index_key(), device.device_id)
        await pipe.execute()

    async def load_devices(self) -> list[DeviceNode]:
        """Load all registered devices."""
        redis = await self._get_redis()

        dev_ids = await redis.smembers(self._dev_index_key())
        if not dev_ids:
            return []

        keys = [self._dev_key(did.decode() if isinstance(did, bytes) else did) for did in dev_ids]
        data_list = await redis.mget(keys)

        devices = []
        for data in data_list:
            if data:
                try:
                    device = _dict_to_device(json.loads(data))
                    devices.append(device)
                except Exception as e:
                    logger.warning(f"Failed to parse device: {e}")
        return devices

    async def delete_device(self, device_id: str) -> bool:
        """Delete a device registration."""
        redis = await self._get_redis()
        key = self._dev_key(device_id)

        pipe = redis.pipeline()
        pipe.delete(key)
        pipe.srem(self._dev_index_key(), device_id)
        results = await pipe.execute()
        return results[0] > 0

    async def save_rule(self, rule: RoutingRule) -> None:
        """Save a routing rule."""
        redis = await self._get_redis()
        key = self._rule_key(rule.rule_id)
        data = json.dumps(_rule_to_dict(rule))

        pipe = redis.pipeline()
        pipe.set(key, data)
        pipe.zadd(self._rule_index_key(), {rule.rule_id: rule.priority})
        await pipe.execute()

    async def load_rules(self) -> list[RoutingRule]:
        """Load all routing rules."""
        redis = await self._get_redis()

        # Get rule IDs sorted by priority (highest first)
        rule_ids = await redis.zrevrange(self._rule_index_key(), 0, -1)
        if not rule_ids:
            return []

        keys = [self._rule_key(rid.decode() if isinstance(rid, bytes) else rid) for rid in rule_ids]
        data_list = await redis.mget(keys)

        rules = []
        for data in data_list:
            if data:
                try:
                    rule = _dict_to_rule(json.loads(data))
                    rules.append(rule)
                except Exception as e:
                    logger.warning(f"Failed to parse rule: {e}")
        return rules

    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a routing rule."""
        redis = await self._get_redis()
        key = self._rule_key(rule_id)

        pipe = redis.pipeline()
        pipe.delete(key)
        pipe.zrem(self._rule_index_key(), rule_id)
        results = await pipe.execute()
        return results[0] > 0

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# =============================================================================
# Factory Function
# =============================================================================


def get_gateway_store(
    backend: str = "auto",
    **kwargs: Any,
) -> GatewayStore:
    """
    Get a gateway store instance.

    Args:
        backend: One of "memory", "file", "redis", or "auto".
                 "auto" will try Redis first, then fall back to file.
        **kwargs: Backend-specific configuration.

    Returns:
        A GatewayStore instance.
    """
    if backend == "memory":
        return InMemoryGatewayStore()
    elif backend == "file":
        return FileGatewayStore(**kwargs)
    elif backend == "redis":
        return RedisGatewayStore(**kwargs)
    elif backend == "auto":
        # Try Redis if URL is provided or env var is set
        redis_url = kwargs.get("redis_url") or os.environ.get("REDIS_URL")
        if redis_url:
            try:
                import redis.asyncio as aioredis  # noqa: F401

                return RedisGatewayStore(redis_url=redis_url, **kwargs)
            except ImportError:
                logger.info("Redis not available, falling back to file store")

        # Fall back to file store
        return FileGatewayStore(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
