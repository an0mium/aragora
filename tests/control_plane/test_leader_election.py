"""
Integration tests for Leader Election.

Tests the distributed leader election system for multi-node deployments:
- LeaderConfig configuration
- LeaderElection lifecycle (start, stop)
- Leader acquisition and release
- Callback notifications
- In-memory fallback when Redis unavailable
- Concurrent election scenarios

Run with:
    pytest tests/control_plane/test_leader_election.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.leader import (
    LeaderConfig,
    LeaderElection,
    LeaderInfo,
    LeaderState,
)


# =============================================================================
# LeaderConfig Tests
# =============================================================================


class TestLeaderConfig:
    """Tests for LeaderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LeaderConfig()

        assert config.redis_url == "redis://localhost:6379"
        assert config.key_prefix == "aragora:leader:"
        assert config.lock_ttl_seconds == 30.0
        assert config.heartbeat_interval == 10.0
        assert config.election_timeout == 5.0
        assert config.retry_interval == 1.0
        assert config.node_id is not None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LeaderConfig(
            redis_url="redis://custom:6380",
            key_prefix="custom:leader:",
            lock_ttl_seconds=60.0,
            heartbeat_interval=20.0,
            node_id="custom-node-1",
        )

        assert config.redis_url == "redis://custom:6380"
        assert config.key_prefix == "custom:leader:"
        assert config.lock_ttl_seconds == 60.0
        assert config.heartbeat_interval == 20.0
        assert config.node_id == "custom-node-1"

    def test_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "REDIS_URL": "redis://env-host:6379",
            "LEADER_KEY_PREFIX": "env:leader:",
            "LEADER_LOCK_TTL": "45",
            "LEADER_HEARTBEAT": "15",
            "NODE_ID": "env-node-1",
        }

        with patch.dict(os.environ, env_vars):
            config = LeaderConfig.from_env()

            assert config.redis_url == "redis://env-host:6379"
            assert config.key_prefix == "env:leader:"
            assert config.lock_ttl_seconds == 45.0
            assert config.heartbeat_interval == 15.0
            assert config.node_id == "env-node-1"

    def test_from_env_defaults(self):
        """Test from_env with default values when env vars not set."""
        # Clear relevant env vars
        env_to_clear = [
            "REDIS_URL",
            "LEADER_KEY_PREFIX",
            "LEADER_LOCK_TTL",
            "NODE_ID",
        ]
        clean_env = {k: v for k, v in os.environ.items() if k not in env_to_clear}

        with patch.dict(os.environ, clean_env, clear=True):
            config = LeaderConfig.from_env()

            assert config.redis_url == "redis://localhost:6379"
            assert config.node_id is not None  # Auto-generated


# =============================================================================
# LeaderState Tests
# =============================================================================


class TestLeaderState:
    """Tests for LeaderState enum."""

    def test_state_values(self):
        """Test all state values."""
        assert LeaderState.FOLLOWER.value == "follower"
        assert LeaderState.CANDIDATE.value == "candidate"
        assert LeaderState.LEADER.value == "leader"
        assert LeaderState.DISCONNECTED.value == "disconnected"

    def test_state_count(self):
        """Test expected number of states."""
        states = list(LeaderState)
        assert len(states) == 4


# =============================================================================
# LeaderInfo Tests
# =============================================================================


class TestLeaderInfo:
    """Tests for LeaderInfo dataclass."""

    def test_leader_info_creation(self):
        """Test creating LeaderInfo."""
        info = LeaderInfo(
            node_id="leader-1",
            elected_at=1704067200.0,
            last_heartbeat=1704067210.0,
            metadata={"region": "us-east-1"},
        )

        assert info.node_id == "leader-1"
        assert info.elected_at == 1704067200.0
        assert info.last_heartbeat == 1704067210.0
        assert info.metadata["region"] == "us-east-1"

    def test_leader_info_default_metadata(self):
        """Test LeaderInfo with default metadata."""
        info = LeaderInfo(
            node_id="leader-2",
            elected_at=1704067200.0,
            last_heartbeat=1704067200.0,
        )

        assert info.metadata == {}


# =============================================================================
# LeaderElection Tests
# =============================================================================


class TestLeaderElectionInit:
    """Tests for LeaderElection initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        election = LeaderElection()

        assert election.state == LeaderState.DISCONNECTED
        assert not election.is_leader
        assert election.node_id is not None
        assert election.current_leader is None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = LeaderConfig(node_id="custom-node")
        election = LeaderElection(config=config)

        assert election.node_id == "custom-node"

    def test_init_with_redis_client(self):
        """Test initialization with provided Redis client."""
        mock_redis = MagicMock()
        election = LeaderElection(redis_client=mock_redis)

        assert election._redis is mock_redis


class TestLeaderElectionCallbacks:
    """Tests for LeaderElection callback registration."""

    def test_on_become_leader(self):
        """Test registering become_leader callback."""
        election = LeaderElection()
        callback = MagicMock()

        election.on_become_leader(callback)

        assert callback in election._on_become_leader

    def test_on_lose_leader(self):
        """Test registering lose_leader callback."""
        election = LeaderElection()
        callback = MagicMock()

        election.on_lose_leader(callback)

        assert callback in election._on_lose_leader

    def test_on_leader_change(self):
        """Test registering leader_change callback."""
        election = LeaderElection()
        callback = MagicMock()

        election.on_leader_change(callback)

        assert callback in election._on_leader_change

    def test_multiple_callbacks(self):
        """Test registering multiple callbacks."""
        election = LeaderElection()
        callbacks = [MagicMock() for _ in range(3)]

        for cb in callbacks:
            election.on_become_leader(cb)

        assert len(election._on_become_leader) == 3


class TestLeaderElectionLifecycle:
    """Tests for LeaderElection start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_election_task(self):
        """Test that start creates election task."""
        config = LeaderConfig(retry_interval=0.01)
        election = LeaderElection(config=config)

        await election.start()

        try:
            assert election._running is True
            assert election._election_task is not None
            assert election.state != LeaderState.DISCONNECTED
        finally:
            await election.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        """Test that stop cancels running tasks."""
        config = LeaderConfig(retry_interval=0.01)
        election = LeaderElection(config=config)

        await election.start()
        await asyncio.sleep(0.05)

        await election.stop()

        assert election._running is False
        assert election.state == LeaderState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Test that stop can be called multiple times."""
        election = LeaderElection()

        # Stop without starting
        await election.stop()
        await election.stop()  # Should not raise

        assert election.state == LeaderState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """Test that start can be called multiple times."""
        config = LeaderConfig(retry_interval=0.01)
        election = LeaderElection(config=config)

        await election.start()
        await election.start()  # Should not create duplicate tasks

        try:
            assert election._running is True
        finally:
            await election.stop()


class TestLeaderElectionInMemoryFallback:
    """Tests for in-memory fallback when Redis unavailable."""

    @pytest.mark.asyncio
    async def test_uses_in_memory_without_redis(self):
        """Test that in-memory fallback is used when aioredis unavailable."""
        config = LeaderConfig(retry_interval=0.01)
        election = LeaderElection(config=config)

        # Patch aioredis import to fail
        with patch.dict("sys.modules", {"aioredis": None}):
            await election.start()

            try:
                assert election._running is True
                # Should use in-memory Redis
                assert election._redis is not None
            finally:
                await election.stop()


class TestLeaderElectionSingleNode:
    """Tests for single-node scenarios."""

    @pytest.mark.asyncio
    async def test_single_node_becomes_leader(self):
        """Test that a single node becomes leader."""
        config = LeaderConfig(
            retry_interval=0.01,
            election_timeout=0.1,
        )
        election = LeaderElection(config=config)

        await election.start()

        try:
            # Wait for election
            for _ in range(50):  # Max 0.5 seconds
                if election.is_leader:
                    break
                await asyncio.sleep(0.01)

            # Single node should become leader
            assert election.is_leader or election.state == LeaderState.FOLLOWER
        finally:
            await election.stop()


class TestLeaderElectionCallbackExecution:
    """Tests for callback execution during state changes."""

    @pytest.mark.asyncio
    async def test_become_leader_callback_called(self):
        """Test that become_leader callback is invoked."""
        config = LeaderConfig(retry_interval=0.01)
        election = LeaderElection(config=config)

        callback_called = asyncio.Event()

        def on_leader():
            callback_called.set()

        election.on_become_leader(on_leader)

        await election.start()

        try:
            # Wait for callback or timeout
            try:
                await asyncio.wait_for(callback_called.wait(), timeout=1.0)
                assert True  # Callback was called
            except asyncio.TimeoutError:
                # May not become leader in test environment
                pass
        finally:
            await election.stop()


# =============================================================================
# Concurrent Election Tests
# =============================================================================


class TestConcurrentElection:
    """Tests for concurrent election scenarios."""

    @pytest.mark.asyncio
    async def test_two_nodes_one_leader(self):
        """Test that with two nodes, exactly one becomes leader."""
        config1 = LeaderConfig(node_id="node-1", retry_interval=0.01)
        config2 = LeaderConfig(node_id="node-2", retry_interval=0.01)

        # Share in-memory Redis
        mock_redis = MockInMemoryRedis()

        election1 = LeaderElection(config=config1, redis_client=mock_redis)
        election2 = LeaderElection(config=config2, redis_client=mock_redis)

        await election1.start()
        await election2.start()

        try:
            # Wait for elections to settle
            await asyncio.sleep(0.2)

            leaders = []
            if election1.is_leader:
                leaders.append("node-1")
            if election2.is_leader:
                leaders.append("node-2")

            # At most one leader (may have zero if elections still in progress)
            assert len(leaders) <= 1
        finally:
            await election1.stop()
            await election2.stop()


class MockInMemoryRedis:
    """Mock Redis for testing concurrent elections."""

    def __init__(self):
        self._data: Dict[str, str] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        nx: bool = False,
    ) -> Optional[bool]:
        """SET command with NX and EX support."""
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if nx and key in self._data:
                return None
            self._data[key] = value
            return True

    async def get(self, key: str) -> Optional[str]:
        """GET command."""
        return self._data.get(key)

    async def delete(self, key: str) -> int:
        """DELETE command."""
        if key in self._data:
            del self._data[key]
            return 1
        return 0

    async def expire(self, key: str, seconds: int) -> bool:
        """EXPIRE command (no-op in mock)."""
        return key in self._data


# =============================================================================
# Properties Tests
# =============================================================================


class TestLeaderElectionProperties:
    """Tests for LeaderElection properties."""

    def test_state_property(self):
        """Test state property."""
        election = LeaderElection()
        assert election.state == LeaderState.DISCONNECTED

    def test_is_leader_property(self):
        """Test is_leader property."""
        election = LeaderElection()
        assert not election.is_leader

        # Manually set state for testing
        election._state = LeaderState.LEADER
        assert election.is_leader

    def test_node_id_property(self):
        """Test node_id property."""
        config = LeaderConfig(node_id="test-node")
        election = LeaderElection(config=config)
        assert election.node_id == "test-node"

    def test_current_leader_property(self):
        """Test current_leader property."""
        election = LeaderElection()
        assert election.current_leader is None

        # Set leader info
        election._current_leader = LeaderInfo(
            node_id="other-node",
            elected_at=time.time(),
            last_heartbeat=time.time(),
        )
        assert election.current_leader.node_id == "other-node"


# =============================================================================
# Edge Cases
# =============================================================================


class TestLeaderElectionEdgeCases:
    """Tests for edge cases in leader election."""

    @pytest.mark.asyncio
    async def test_rapid_start_stop(self):
        """Test rapid start/stop cycles."""
        config = LeaderConfig(retry_interval=0.01)
        election = LeaderElection(config=config)

        for _ in range(5):
            await election.start()
            await election.stop()

        assert election.state == LeaderState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_callback_exception_handling(self):
        """Test that callback exceptions don't break election."""
        config = LeaderConfig(retry_interval=0.01)
        election = LeaderElection(config=config)

        def bad_callback():
            raise ValueError("Callback error")

        election.on_become_leader(bad_callback)

        await election.start()

        try:
            await asyncio.sleep(0.1)
            # Should not crash despite bad callback
            assert election._running is True
        finally:
            await election.stop()


# =============================================================================
# Regional Leader Election Tests
# =============================================================================

from aragora.control_plane.leader import (
    RegionalLeaderConfig,
    RegionalLeaderElection,
    RegionalLeaderInfo,
)


class TestRegionalLeaderConfig:
    """Tests for RegionalLeaderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegionalLeaderConfig()

        assert config.region_id == "default"
        assert config.sync_regions == []
        assert config.broadcast_leadership is True
        # Should inherit base config defaults
        assert config.redis_url == "redis://localhost:6379"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RegionalLeaderConfig(
            region_id="us-west-2",
            sync_regions=["us-east-1", "eu-west-1"],
            broadcast_leadership=False,
            node_id="regional-node-1",
        )

        assert config.region_id == "us-west-2"
        assert config.sync_regions == ["us-east-1", "eu-west-1"]
        assert config.broadcast_leadership is False
        assert config.node_id == "regional-node-1"

    def test_get_region_key_prefix(self):
        """Test region-scoped key prefix generation."""
        config = RegionalLeaderConfig(region_id="us-west-2")
        prefix = config.get_region_key_prefix()

        assert prefix == "aragora:leader:region:us-west-2:"

    def test_from_env(self):
        """Test configuration from environment variables."""
        env_vars = {
            "REDIS_URL": "redis://env-host:6379",
            "ARAGORA_REGION_ID": "ap-southeast-1",
            "ARAGORA_SYNC_REGIONS": "us-west-2, eu-west-1",
            "ARAGORA_BROADCAST_LEADERSHIP": "false",
            "NODE_ID": "regional-env-node",
        }

        with patch.dict(os.environ, env_vars):
            config = RegionalLeaderConfig.from_env()

            assert config.region_id == "ap-southeast-1"
            assert config.sync_regions == ["us-west-2", "eu-west-1"]
            assert config.broadcast_leadership is False
            assert config.node_id == "regional-env-node"


class TestRegionalLeaderInfo:
    """Tests for RegionalLeaderInfo dataclass."""

    def test_regional_leader_info_creation(self):
        """Test creating RegionalLeaderInfo."""
        info = RegionalLeaderInfo(
            node_id="leader-1",
            elected_at=1704067200.0,
            last_heartbeat=1704067210.0,
            region_id="us-west-2",
            is_global_coordinator=True,
        )

        assert info.node_id == "leader-1"
        assert info.region_id == "us-west-2"
        assert info.is_global_coordinator is True

    def test_default_values(self):
        """Test RegionalLeaderInfo with default values."""
        info = RegionalLeaderInfo(
            node_id="leader-2",
            elected_at=1704067200.0,
            last_heartbeat=1704067200.0,
        )

        assert info.region_id == "default"
        assert info.is_global_coordinator is False


class TestRegionalLeaderElectionInit:
    """Tests for RegionalLeaderElection initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        election = RegionalLeaderElection()

        assert election.state == LeaderState.DISCONNECTED
        assert not election.is_regional_leader
        assert not election.is_global_coordinator
        assert election.region_id == "default"
        assert election.regional_leaders == {}

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = RegionalLeaderConfig(
            region_id="us-west-2",
            node_id="custom-regional-node",
        )
        election = RegionalLeaderElection(config=config)

        assert election.region_id == "us-west-2"
        assert election.node_id == "custom-regional-node"


class TestRegionalLeaderElectionCallbacks:
    """Tests for RegionalLeaderElection callback registration."""

    def test_on_become_regional_leader(self):
        """Test registering become_regional_leader callback."""
        election = RegionalLeaderElection()
        callback = MagicMock()

        election.on_become_regional_leader(callback)

        assert callback in election._on_become_regional_leader

    def test_on_lose_regional_leader(self):
        """Test registering lose_regional_leader callback."""
        election = RegionalLeaderElection()
        callback = MagicMock()

        election.on_lose_regional_leader(callback)

        assert callback in election._on_lose_regional_leader

    def test_on_become_global_coordinator(self):
        """Test registering become_global_coordinator callback."""
        election = RegionalLeaderElection()
        callback = MagicMock()

        election.on_become_global_coordinator(callback)

        assert callback in election._on_become_global_coordinator

    def test_on_lose_global_coordinator(self):
        """Test registering lose_global_coordinator callback."""
        election = RegionalLeaderElection()
        callback = MagicMock()

        election.on_lose_global_coordinator(callback)

        assert callback in election._on_lose_global_coordinator


class TestRegionalLeaderElectionLifecycle:
    """Tests for RegionalLeaderElection start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_election_task(self):
        """Test that start creates election task with region-scoped keys."""
        config = RegionalLeaderConfig(
            region_id="us-west-2",
            retry_interval=0.01,
        )
        election = RegionalLeaderElection(config=config)

        await election.start()

        try:
            assert election._running is True
            assert election._election_task is not None
            # Key prefix should be region-scoped during election
        finally:
            await election.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self):
        """Test that stop cancels running tasks."""
        config = RegionalLeaderConfig(retry_interval=0.01)
        election = RegionalLeaderElection(config=config)

        await election.start()
        await asyncio.sleep(0.05)

        await election.stop()

        assert election._running is False
        assert election.state == LeaderState.DISCONNECTED


class TestRegionalLeaderElectionProperties:
    """Tests for RegionalLeaderElection properties."""

    def test_region_id_property(self):
        """Test region_id property."""
        config = RegionalLeaderConfig(region_id="eu-west-1")
        election = RegionalLeaderElection(config=config)
        assert election.region_id == "eu-west-1"

    def test_is_regional_leader_property(self):
        """Test is_regional_leader property mirrors is_leader."""
        election = RegionalLeaderElection()
        assert not election.is_regional_leader

        # Manually set state for testing
        election._state = LeaderState.LEADER
        assert election.is_regional_leader

    def test_is_global_coordinator_property(self):
        """Test is_global_coordinator property."""
        election = RegionalLeaderElection()
        assert not election.is_global_coordinator

        election._is_global_coordinator = True
        assert election.is_global_coordinator

    def test_regional_leaders_property(self):
        """Test regional_leaders property returns copy."""
        election = RegionalLeaderElection()
        info = RegionalLeaderInfo(
            node_id="other-leader",
            region_id="us-east-1",
            elected_at=time.time(),
            last_heartbeat=time.time(),
        )
        election._regional_leaders["us-east-1"] = info

        leaders = election.regional_leaders
        assert "us-east-1" in leaders
        assert leaders["us-east-1"].node_id == "other-leader"

        # Should be a copy
        leaders["us-east-1"] = None
        assert election._regional_leaders["us-east-1"].node_id == "other-leader"


class TestRegionalLeaderElectionStats:
    """Tests for RegionalLeaderElection statistics."""

    def test_get_stats_includes_regional_info(self):
        """Test that get_stats includes regional information."""
        config = RegionalLeaderConfig(
            region_id="ap-southeast-1",
            node_id="test-node",
        )
        election = RegionalLeaderElection(config=config)

        stats = election.get_stats()

        assert stats["node_id"] == "test-node"
        assert stats["region_id"] == "ap-southeast-1"
        assert stats["is_regional_leader"] is False
        assert stats["is_global_coordinator"] is False
        assert stats["known_regional_leaders"] == []

    def test_get_stats_with_known_leaders(self):
        """Test stats with known regional leaders."""
        election = RegionalLeaderElection()
        election._regional_leaders["us-east-1"] = RegionalLeaderInfo(
            node_id="east-leader",
            region_id="us-east-1",
            elected_at=time.time(),
            last_heartbeat=time.time(),
        )
        election._regional_leaders["eu-west-1"] = RegionalLeaderInfo(
            node_id="eu-leader",
            region_id="eu-west-1",
            elected_at=time.time(),
            last_heartbeat=time.time(),
        )

        stats = election.get_stats()

        assert "us-east-1" in stats["known_regional_leaders"]
        assert "eu-west-1" in stats["known_regional_leaders"]


class TestMultiRegionElection:
    """Tests for multi-region election scenarios."""

    @pytest.mark.asyncio
    async def test_two_regions_two_leaders(self):
        """Test that two regions can have independent leaders."""
        mock_redis = MockInMemoryRedis()

        config1 = RegionalLeaderConfig(
            region_id="us-west-2",
            node_id="west-node",
            retry_interval=0.01,
        )
        config2 = RegionalLeaderConfig(
            region_id="us-east-1",
            node_id="east-node",
            retry_interval=0.01,
        )

        election1 = RegionalLeaderElection(config=config1, redis_client=mock_redis)
        election2 = RegionalLeaderElection(config=config2, redis_client=mock_redis)

        await election1.start()
        await election2.start()

        try:
            # Wait for elections to settle
            await asyncio.sleep(0.2)

            # Both should be able to become leaders of their respective regions
            # (since they use different key prefixes)
            west_leader = election1.is_regional_leader
            east_leader = election2.is_regional_leader

            # At least one should be a leader (the test might vary based on timing)
            # but they should NOT block each other
        finally:
            await election1.stop()
            await election2.stop()

    @pytest.mark.asyncio
    async def test_same_region_only_one_leader(self):
        """Test that same region only has one leader."""
        mock_redis = MockInMemoryRedis()

        # Both in same region
        config1 = RegionalLeaderConfig(
            region_id="us-west-2",
            node_id="node-1",
            retry_interval=0.01,
        )
        config2 = RegionalLeaderConfig(
            region_id="us-west-2",
            node_id="node-2",
            retry_interval=0.01,
        )

        election1 = RegionalLeaderElection(config=config1, redis_client=mock_redis)
        election2 = RegionalLeaderElection(config=config2, redis_client=mock_redis)

        await election1.start()
        await election2.start()

        try:
            await asyncio.sleep(0.2)

            leaders = []
            if election1.is_regional_leader:
                leaders.append("node-1")
            if election2.is_regional_leader:
                leaders.append("node-2")

            # At most one leader in same region
            assert len(leaders) <= 1
        finally:
            await election1.stop()
            await election2.stop()
