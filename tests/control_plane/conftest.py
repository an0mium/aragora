"""
Control Plane test fixtures.

Provides fixtures specific to testing the control plane components:
- ControlPlaneCoordinator
- AgentRegistry
- TaskScheduler
- HealthMonitor
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Mock Redis for Control Plane
# ============================================================================


class MockRedis:
    """
    Mock Redis client for control plane testing.

    Implements Redis operations needed by AgentRegistry and TaskScheduler.
    """

    def __init__(self) -> None:
        self._data: Dict[str, str] = {}
        self._streams: Dict[str, List[tuple]] = {}
        self._consumer_groups: Dict[str, Dict[str, Any]] = {}
        self._message_id_counter = 0

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self._data.clear()
        self._streams.clear()

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
    ) -> bool:
        self._data[key] = value
        return True

    async def delete(self, key: str) -> int:
        if key in self._data:
            del self._data[key]
            return 1
        return 0

    async def scan_iter(self, match: str = "*") -> AsyncGenerator[str, None]:
        """Iterate through keys matching pattern."""
        import fnmatch

        for key in list(self._data.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    async def xgroup_create(
        self,
        stream_key: str,
        group_name: str,
        id: str = "0",
        mkstream: bool = False,
    ) -> bool:
        if stream_key not in self._streams:
            if mkstream:
                self._streams[stream_key] = []
            else:
                raise Exception("Stream does not exist")

        if stream_key not in self._consumer_groups:
            self._consumer_groups[stream_key] = {}
        self._consumer_groups[stream_key][group_name] = {"pending": []}
        return True

    async def xadd(
        self,
        stream_key: str,
        data: Dict[str, str],
    ) -> str:
        if stream_key not in self._streams:
            self._streams[stream_key] = []

        self._message_id_counter += 1
        msg_id = f"{self._message_id_counter}-0"
        self._streams[stream_key].append((msg_id, data))
        return msg_id

    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: Dict[str, str],
        count: int = 1,
        block: int = 0,
    ) -> List[tuple]:
        results = []
        for stream_key, start_id in streams.items():
            if stream_key in self._streams:
                stream_msgs = self._streams[stream_key]
                if stream_msgs:
                    msg = stream_msgs.pop(0)
                    results.append((stream_key, [msg]))
        return results

    async def xack(
        self,
        stream_key: str,
        group_name: str,
        message_id: str,
    ) -> int:
        return 1

    async def xpending_range(
        self,
        stream_key: str,
        group_name: str,
        min: str = "-",
        max: str = "+",
        count: int = 100,
    ) -> List[Dict[str, Any]]:
        return []


@pytest.fixture
def mock_redis() -> MockRedis:
    """Provide a mock Redis client."""
    return MockRedis()


# ============================================================================
# Control Plane Component Fixtures
# ============================================================================


@pytest.fixture
def mock_registry(mock_redis: MockRedis):
    """Create a mock AgentRegistry with in-memory storage."""
    from aragora.control_plane.registry import AgentRegistry

    registry = AgentRegistry(
        redis_url="redis://localhost:6379",
        key_prefix="test:agents:",
        heartbeat_timeout=30.0,
        cleanup_interval=60.0,
    )
    # Use in-memory storage (no Redis connection)
    registry._redis = None
    registry._local_cache = {}
    return registry


@pytest.fixture
def mock_scheduler(mock_redis: MockRedis):
    """Create a mock TaskScheduler with in-memory storage."""
    from aragora.control_plane.scheduler import TaskScheduler

    scheduler = TaskScheduler(
        redis_url="redis://localhost:6379",
        key_prefix="test:tasks:",
        stream_prefix="test:stream:",
    )
    # Use in-memory storage (no Redis connection)
    scheduler._redis = None
    scheduler._local_tasks = {}
    scheduler._local_queue = []
    return scheduler


@pytest.fixture
def mock_health_monitor(mock_registry):
    """Create a mock HealthMonitor."""
    from aragora.control_plane.health import HealthMonitor

    monitor = HealthMonitor(
        registry=mock_registry,
        probe_interval=30.0,
        probe_timeout=10.0,
        unhealthy_threshold=3,
        recovery_threshold=2,
    )
    return monitor


@pytest.fixture
async def coordinator(mock_registry, mock_scheduler, mock_health_monitor):
    """
    Create a ControlPlaneCoordinator with mock components.

    This fixture provides a fully configured coordinator for testing
    without requiring Redis.
    """
    from aragora.control_plane.coordinator import (
        ControlPlaneConfig,
        ControlPlaneCoordinator,
    )

    config = ControlPlaneConfig(
        redis_url="redis://localhost:6379",
        key_prefix="test:cp:",
        heartbeat_timeout=30.0,
        task_timeout=60.0,
        max_task_retries=3,
    )

    coordinator = ControlPlaneCoordinator(
        config=config,
        registry=mock_registry,
        scheduler=mock_scheduler,
        health_monitor=mock_health_monitor,
    )

    # Mark as connected without actually connecting
    coordinator._connected = True

    yield coordinator

    # Cleanup
    coordinator._connected = False


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_agent() -> Dict[str, Any]:
    """Sample agent registration data."""
    return {
        "agent_id": "test-claude",
        "capabilities": ["debate", "code", "analysis"],
        "model": "claude-3-opus",
        "provider": "anthropic",
        "metadata": {"version": "1.0"},
    }


@pytest.fixture
def sample_agents() -> List[Dict[str, Any]]:
    """Multiple agent configurations for testing."""
    return [
        {
            "agent_id": "agent-claude",
            "capabilities": ["debate", "code"],
            "model": "claude-3-opus",
            "provider": "anthropic",
        },
        {
            "agent_id": "agent-gpt",
            "capabilities": ["debate", "analysis"],
            "model": "gpt-4",
            "provider": "openai",
        },
        {
            "agent_id": "agent-gemini",
            "capabilities": ["code", "research"],
            "model": "gemini-pro",
            "provider": "google",
        },
    ]


@pytest.fixture
def sample_task() -> Dict[str, Any]:
    """Sample task data."""
    return {
        "task_type": "debate",
        "payload": {"question": "What is the best approach to testing?"},
        "required_capabilities": ["debate"],
    }


# ============================================================================
# Async Test Helpers
# ============================================================================


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
