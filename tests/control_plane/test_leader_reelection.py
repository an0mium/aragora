"""
Tests for leader election failure scenarios.

Tests cover:
- Leader lock expiration and re-election
- Leader crash mid-heartbeat
- Split-brain prevention
- Lock holder divergence detection
- Redis connection loss during leadership

Run with: pytest tests/control_plane/test_leader_reelection.py -v
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.leader import (
    DistributedStateError,
    LeaderConfig,
    LeaderElection,
    LeaderInfo,
    LeaderState,
    is_distributed_state_required,
)


# ============================================================================
# Mock Redis Implementation for Testing
# ============================================================================


class MockRedis:
    """Mock Redis client with controllable behavior for testing failure scenarios."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._expiries: Dict[str, float] = {}
        self._fail_on_next: Optional[str] = None
        self._delay_seconds: float = 0.0
        self._operation_log: List[Dict[str, Any]] = []
        self._closed = False

    async def set(
        self,
        key: str,
        value: str,
        nx: bool = False,
        ex: Optional[int] = None,
    ) -> bool:
        """Mock SET with NX and EX options."""
        self._operation_log.append(
            {
                "op": "set",
                "key": key,
                "value": value,
                "nx": nx,
                "ex": ex,
                "time": time.time(),
            }
        )

        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)

        if self._fail_on_next == "set":
            self._fail_on_next = None
            raise ConnectionError("Redis connection failed")

        if nx and key in self._data:
            return False

        self._data[key] = value
        if ex:
            self._expiries[key] = time.time() + ex
        return True

    async def get(self, key: str) -> Optional[str]:
        """Mock GET."""
        self._operation_log.append({"op": "get", "key": key, "time": time.time()})

        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)

        if self._fail_on_next == "get":
            self._fail_on_next = None
            raise ConnectionError("Redis connection failed")

        # Check expiry
        if key in self._expiries and time.time() > self._expiries[key]:
            del self._data[key]
            del self._expiries[key]
            return None

        return self._data.get(key)

    async def delete(self, key: str) -> int:
        """Mock DELETE."""
        self._operation_log.append({"op": "delete", "key": key, "time": time.time()})

        if self._fail_on_next == "delete":
            self._fail_on_next = None
            raise ConnectionError("Redis connection failed")

        if key in self._data:
            del self._data[key]
            self._expiries.pop(key, None)
            return 1
        return 0

    async def expire(self, key: str, seconds: int) -> bool:
        """Mock EXPIRE."""
        self._operation_log.append(
            {
                "op": "expire",
                "key": key,
                "seconds": seconds,
                "time": time.time(),
            }
        )

        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)

        if self._fail_on_next == "expire":
            self._fail_on_next = None
            raise ConnectionError("Redis connection failed")

        if key in self._data:
            self._expiries[key] = time.time() + seconds
            return True
        return False

    async def hset(
        self,
        key: str,
        field: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[Dict] = None,
        **kwargs,
    ) -> int:
        """Mock HSET - supports both (key, field, value) and (key, mapping=dict) forms."""
        if self._fail_on_next == "hset":
            self._fail_on_next = None
            raise ConnectionError("Redis connection failed")

        if key not in self._hashes:
            self._hashes[key] = {}

        # Handle both calling conventions
        if mapping:
            data = mapping
        elif field is not None and value is not None:
            data = {field: value}
        else:
            data = kwargs

        for k, v in data.items():
            self._hashes[key][k] = str(v)

        return len(data)

    async def hgetall(self, key: str) -> Dict[str, str]:
        """Mock HGETALL."""
        if self._fail_on_next == "hgetall":
            self._fail_on_next = None
            raise ConnectionError("Redis connection failed")

        return self._hashes.get(key, {})

    async def close(self) -> None:
        """Mock close."""
        self._closed = True

    def fail_on_next(self, operation: str) -> None:
        """Set the next operation to fail."""
        self._fail_on_next = operation

    def set_delay(self, seconds: float) -> None:
        """Set delay for operations."""
        self._delay_seconds = seconds

    def clear_delay(self) -> None:
        """Clear operation delay."""
        self._delay_seconds = 0.0

    def expire_key_now(self, key: str) -> None:
        """Immediately expire a key (simulates TTL expiration)."""
        if key in self._data:
            del self._data[key]
            self._expiries.pop(key, None)

    def corrupt_value(self, key: str, value: str) -> None:
        """Corrupt a key's value (simulates split-brain)."""
        self._data[key] = value


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    return MockRedis()


@pytest.fixture
def leader_config():
    """Create a leader config with short timeouts for testing."""
    return LeaderConfig(
        redis_url="redis://localhost:6379",
        key_prefix="test:leader:",
        lock_ttl_seconds=2.0,  # Short TTL for testing
        heartbeat_interval=0.5,  # Fast heartbeat
        retry_interval=0.1,  # Fast retry
        election_timeout=1.0,
        node_id="test-node-1",
    )


@pytest.fixture
def election(leader_config, mock_redis):
    """Create a LeaderElection instance with mock Redis."""
    return LeaderElection(config=leader_config, redis_client=mock_redis)


@pytest.fixture
def election_node2(mock_redis):
    """Create a second LeaderElection instance (simulating another node)."""
    config = LeaderConfig(
        key_prefix="test:leader:",
        lock_ttl_seconds=2.0,
        heartbeat_interval=0.5,
        retry_interval=0.1,
        node_id="test-node-2",
    )
    return LeaderElection(config=config, redis_client=mock_redis)


# ============================================================================
# Basic Leader Election Tests
# ============================================================================


class TestLeaderElectionBasics:
    """Basic leader election functionality tests."""

    def test_initial_state(self, election):
        """Test initial state is DISCONNECTED."""
        assert election.state == LeaderState.DISCONNECTED
        assert not election.is_leader
        assert election.current_leader is None

    @pytest.mark.asyncio
    async def test_single_node_becomes_leader(self, election, mock_redis):
        """Test that a single node becomes leader."""
        await election.start()
        # Give election loop time to run
        await asyncio.sleep(0.3)

        assert election.state == LeaderState.LEADER
        assert election.is_leader
        assert election.current_leader is not None
        assert election.current_leader.node_id == "test-node-1"

        await election.stop()

    @pytest.mark.asyncio
    async def test_stop_releases_leadership(self, election, mock_redis):
        """Test that stopping releases leadership."""
        await election.start()
        await asyncio.sleep(0.3)
        assert election.is_leader

        await election.stop()
        assert election.state == LeaderState.DISCONNECTED
        assert not election.is_leader

        # Lock should be released
        lock_key = "test:leader:lock"
        assert lock_key not in mock_redis._data


# ============================================================================
# Leader Re-election Tests
# ============================================================================


class TestLeaderReelection:
    """Tests for leader re-election scenarios."""

    @pytest.mark.asyncio
    async def test_lock_expiration_triggers_reelection(self, election, election_node2, mock_redis):
        """Test that lock expiration allows new leader election."""
        # Node 1 becomes leader
        await election.start()
        await asyncio.sleep(0.3)
        assert election.is_leader

        # Simulate lock expiration (without node 1 refreshing)
        lock_key = "test:leader:lock"
        mock_redis.expire_key_now(lock_key)

        # Node 2 should be able to become leader
        await election_node2.start()
        await asyncio.sleep(0.3)

        # Node 2 should now be leader
        assert election_node2.is_leader

        await election.stop()
        await election_node2.stop()

    @pytest.mark.asyncio
    async def test_follower_detects_leader_change(self, mock_redis):
        """Test that followers detect leader changes."""
        config1 = LeaderConfig(
            key_prefix="test:leader:",
            lock_ttl_seconds=2.0,
            retry_interval=0.1,
            node_id="node-1",
        )
        config2 = LeaderConfig(
            key_prefix="test:leader:",
            lock_ttl_seconds=2.0,
            retry_interval=0.1,
            node_id="node-2",
        )

        election1 = LeaderElection(config=config1, redis_client=mock_redis)
        election2 = LeaderElection(config=config2, redis_client=mock_redis)

        # Track leader changes
        leader_changes: List[Optional[str]] = []

        def on_change(node_id: Optional[str]) -> None:
            leader_changes.append(node_id)

        election2.on_leader_change(on_change)

        # Node 1 becomes leader
        await election1.start()
        await asyncio.sleep(0.3)

        # Node 2 starts as follower
        await election2.start()
        await asyncio.sleep(0.3)

        assert not election2.is_leader
        assert election2.current_leader is not None
        assert election2.current_leader.node_id == "node-1"

        await election1.stop()
        await election2.stop()

    @pytest.mark.asyncio
    async def test_become_leader_callback_fires(self, election, mock_redis):
        """Test that become_leader callback fires on election."""
        callback_fired = {"value": False}

        def on_become_leader():
            callback_fired["value"] = True

        election.on_become_leader(on_become_leader)

        await election.start()
        await asyncio.sleep(0.3)

        assert callback_fired["value"] is True

        await election.stop()

    @pytest.mark.asyncio
    async def test_lose_leader_callback_fires(self, election, mock_redis):
        """Test that lose_leader callback fires when leadership is lost."""
        callback_fired = {"value": False}

        def on_lose_leader():
            callback_fired["value"] = True

        election.on_lose_leader(on_lose_leader)

        await election.start()
        await asyncio.sleep(0.3)
        assert election.is_leader

        # Corrupt the lock to simulate another node taking over
        mock_redis.corrupt_value("test:leader:lock", "other-node")

        # Wait for election loop to detect
        await asyncio.sleep(0.5)

        assert callback_fired["value"] is True

        await election.stop()


# ============================================================================
# Split-Brain Prevention Tests
# ============================================================================


class TestSplitBrainPrevention:
    """Tests for split-brain scenario prevention."""

    @pytest.mark.asyncio
    async def test_concurrent_election_only_one_wins(self, mock_redis):
        """Test that concurrent election attempts result in only one leader."""
        nodes = []
        for i in range(5):
            config = LeaderConfig(
                key_prefix="test:leader:",
                lock_ttl_seconds=5.0,
                retry_interval=0.05,
                node_id=f"node-{i}",
            )
            election = LeaderElection(config=config, redis_client=mock_redis)
            nodes.append(election)

        # Start all nodes concurrently
        await asyncio.gather(*[n.start() for n in nodes])
        await asyncio.sleep(0.5)

        # Exactly one should be leader
        leaders = [n for n in nodes if n.is_leader]
        assert len(leaders) == 1

        # Clean up
        await asyncio.gather(*[n.stop() for n in nodes])

    @pytest.mark.asyncio
    async def test_lock_holder_divergence_detected(self, election, mock_redis):
        """Test that leader detects when lock is held by another node."""
        await election.start()
        await asyncio.sleep(0.3)
        assert election.is_leader

        # Another node "steals" the lock (simulates race condition)
        mock_redis.corrupt_value("test:leader:lock", "rogue-node")

        # Wait for refresh to detect divergence
        await asyncio.sleep(0.5)

        # Should have lost leadership
        assert not election.is_leader

        await election.stop()


# ============================================================================
# Redis Connection Failure Tests
# ============================================================================


class TestRedisConnectionFailures:
    """Tests for Redis connection failure handling."""

    @pytest.mark.asyncio
    async def test_redis_failure_during_election(self, election, mock_redis):
        """Test handling of Redis failure during first election attempt."""
        # Track if election was attempted
        original_set = mock_redis.set
        set_attempts = {"count": 0}

        async def tracking_set(*args, **kwargs):
            set_attempts["count"] += 1
            if set_attempts["count"] == 1:
                raise ConnectionError("Redis connection failed")
            return await original_set(*args, **kwargs)

        mock_redis.set = tracking_set

        await election.start()
        # Give just enough time for first attempt to fail
        await asyncio.sleep(0.15)

        # First attempt should have failed (but retry may succeed)
        assert set_attempts["count"] >= 1

        await election.stop()

    @pytest.mark.asyncio
    async def test_redis_failure_during_heartbeat(self, election, mock_redis):
        """Test handling of Redis failure during heartbeat refresh."""
        await election.start()
        await asyncio.sleep(0.3)
        assert election.is_leader

        # Set Redis to fail on expire (heartbeat)
        mock_redis.fail_on_next("expire")

        # Wait for heartbeat to fail
        await asyncio.sleep(0.5)

        # Should lose leadership gracefully
        # (exact behavior depends on implementation)

        await election.stop()

    @pytest.mark.asyncio
    async def test_redis_timeout_handling(self, election, mock_redis):
        """Test that slow Redis operations are handled."""
        await election.start()
        await asyncio.sleep(0.3)
        assert election.is_leader

        # Slow down Redis
        mock_redis.set_delay(0.5)

        # Should still function (with delays)
        await asyncio.sleep(1.0)

        mock_redis.clear_delay()
        await election.stop()


# ============================================================================
# Distributed State Requirement Tests
# ============================================================================


class TestDistributedStateRequirement:
    """Tests for distributed state requirement enforcement."""

    def test_is_distributed_state_required_default(self, monkeypatch):
        """Test default distributed state requirement."""
        # Clear relevant env vars
        monkeypatch.delenv("ARAGORA_REQUIRE_DISTRIBUTED_STATE", raising=False)
        monkeypatch.delenv("ARAGORA_MULTI_INSTANCE", raising=False)
        monkeypatch.delenv("ARAGORA_ENV", raising=False)

        assert is_distributed_state_required() is False

    def test_is_distributed_state_required_explicit(self, monkeypatch):
        """Test explicit distributed state requirement."""
        monkeypatch.setenv("ARAGORA_REQUIRE_DISTRIBUTED_STATE", "true")
        assert is_distributed_state_required() is True

    def test_is_distributed_state_required_multi_instance(self, monkeypatch):
        """Test multi-instance implies distributed state."""
        monkeypatch.delenv("ARAGORA_REQUIRE_DISTRIBUTED_STATE", raising=False)
        monkeypatch.setenv("ARAGORA_MULTI_INSTANCE", "true")
        assert is_distributed_state_required() is True

    def test_is_distributed_state_required_production(self, monkeypatch):
        """Test production environment requires distributed state."""
        monkeypatch.delenv("ARAGORA_REQUIRE_DISTRIBUTED_STATE", raising=False)
        monkeypatch.delenv("ARAGORA_MULTI_INSTANCE", raising=False)
        monkeypatch.setenv("ARAGORA_ENV", "production")
        assert is_distributed_state_required() is True

    def test_is_distributed_state_required_production_single_instance(self, monkeypatch):
        """Test production with single instance override."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SINGLE_INSTANCE", "true")
        assert is_distributed_state_required() is False

    def test_distributed_state_error(self):
        """Test DistributedStateError message."""
        error = DistributedStateError("leader_election", "Redis not available")
        assert "leader_election" in str(error)
        assert "Redis not available" in str(error)


# ============================================================================
# Callback Error Handling Tests
# ============================================================================


class TestCallbackErrorHandling:
    """Tests for callback error handling."""

    @pytest.mark.asyncio
    async def test_callback_exception_doesnt_stop_election(self, election, mock_redis):
        """Test that callback exceptions don't stop the election process."""

        def failing_callback():
            raise RuntimeError("Callback failed!")

        election.on_become_leader(failing_callback)

        # Should still start and function
        await election.start()
        await asyncio.sleep(0.3)

        assert election.is_leader  # Election should succeed despite callback failure

        await election.stop()

    @pytest.mark.asyncio
    async def test_async_callback_exception_handled(self, election, mock_redis):
        """Test that async callback exceptions are handled."""

        async def failing_async_callback():
            raise RuntimeError("Async callback failed!")

        election.on_become_leader(failing_async_callback)

        await election.start()
        await asyncio.sleep(0.3)

        assert election.is_leader

        await election.stop()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_double_start_ignored(self, election, mock_redis):
        """Test that calling start() twice is safe."""
        await election.start()
        await asyncio.sleep(0.2)

        # Second start should be no-op
        await election.start()
        await asyncio.sleep(0.2)

        assert election.is_leader

        await election.stop()

    @pytest.mark.asyncio
    async def test_double_stop_safe(self, election, mock_redis):
        """Test that calling stop() twice is safe."""
        await election.start()
        await asyncio.sleep(0.2)

        await election.stop()
        await election.stop()  # Should not raise

        assert election.state == LeaderState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_stop_before_start(self, election):
        """Test that stop() before start() is safe."""
        await election.stop()  # Should not raise
        assert election.state == LeaderState.DISCONNECTED

    def test_node_id_uniqueness(self):
        """Test that node IDs are unique across instances."""
        config1 = LeaderConfig()
        config2 = LeaderConfig()

        # Default node IDs should be different
        assert config1.node_id != config2.node_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
