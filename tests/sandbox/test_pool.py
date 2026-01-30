"""
Tests for Container Pool Management.

Tests cover:
- ContainerPool lifecycle (start, stop)
- Container acquisition and release
- Pool scaling (up/down)
- Health monitoring
- Session binding
- Pool statistics
- Error handling (pool exhausted, creation failures)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.sandbox.pool import (
    ContainerCreationError,
    ContainerPool,
    ContainerPoolConfig,
    ContainerPoolError,
    ContainerState,
    PooledContainer,
    PoolExhaustedError,
    PoolState,
    PoolStats,
    get_container_pool,
    set_container_pool,
)


class TestContainerPoolConfig:
    """Tests for ContainerPoolConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContainerPoolConfig()

        assert config.min_pool_size == 5
        assert config.max_pool_size == 50
        assert config.warmup_count == 10
        assert config.idle_timeout_seconds == 300.0
        assert config.acquire_timeout_seconds == 30.0
        assert config.creation_timeout_seconds == 60.0
        assert config.health_check_interval_seconds == 30.0
        assert config.max_container_age_seconds == 3600.0
        assert config.base_image == "python:3.11-slim"
        assert config.network_mode == "none"
        assert config.memory_limit_mb == 512
        assert config.cpu_limit == 1.0
        assert config.pids_limit == 100
        assert config.container_prefix == "aragora-sandbox"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContainerPoolConfig(
            min_pool_size=2,
            max_pool_size=10,
            warmup_count=3,
            base_image="node:18-slim",
            memory_limit_mb=256,
        )

        assert config.min_pool_size == 2
        assert config.max_pool_size == 10
        assert config.warmup_count == 3
        assert config.base_image == "node:18-slim"
        assert config.memory_limit_mb == 256

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ContainerPoolConfig(min_pool_size=3, max_pool_size=20)
        data = config.to_dict()

        assert data["min_pool_size"] == 3
        assert data["max_pool_size"] == 20
        assert "base_image" in data
        assert "memory_limit_mb" in data

    def test_config_labels(self):
        """Test configuration with custom labels."""
        config = ContainerPoolConfig(labels={"app": "aragora", "env": "test"})

        assert config.labels["app"] == "aragora"
        assert config.labels["env"] == "test"


class TestPooledContainer:
    """Tests for PooledContainer dataclass."""

    def test_container_creation(self):
        """Test creating a pooled container."""
        container = PooledContainer(
            container_id="abc123",
            container_name="sandbox-test",
            state=ContainerState.READY,
        )

        assert container.container_id == "abc123"
        assert container.container_name == "sandbox-test"
        assert container.state == ContainerState.READY
        assert container.session_id is None
        assert container.execution_count == 0

    def test_container_is_available(self):
        """Test container availability check."""
        container = PooledContainer(
            container_id="abc123",
            container_name="sandbox-test",
            state=ContainerState.READY,
        )

        assert container.is_available() is True

        container.state = ContainerState.ACQUIRED
        assert container.is_available() is False

        container.state = ContainerState.READY
        container.session_id = "session-123"
        assert container.is_available() is False

    def test_container_is_expired_by_age(self):
        """Test container expiration by age."""
        container = PooledContainer(
            container_id="abc123",
            container_name="sandbox-test",
            created_at=time.time() - 3700,  # Over 1 hour ago
            last_used_at=time.time(),
        )

        assert container.is_expired(max_age_seconds=3600, idle_timeout_seconds=300) is True

    def test_container_is_expired_by_idle(self):
        """Test container expiration by idle time."""
        container = PooledContainer(
            container_id="abc123",
            container_name="sandbox-test",
            created_at=time.time(),
            last_used_at=time.time() - 400,  # Over 5 minutes idle
        )

        assert container.is_expired(max_age_seconds=3600, idle_timeout_seconds=300) is True

    def test_container_not_expired(self):
        """Test container that is not expired."""
        now = time.time()
        container = PooledContainer(
            container_id="abc123",
            container_name="sandbox-test",
            created_at=now - 100,
            last_used_at=now - 10,
        )

        assert container.is_expired(max_age_seconds=3600, idle_timeout_seconds=300) is False

    def test_container_to_dict(self):
        """Test container serialization."""
        container = PooledContainer(
            container_id="abc123",
            container_name="sandbox-test",
            state=ContainerState.READY,
        )
        data = container.to_dict()

        assert data["container_id"] == "abc123"
        assert data["container_name"] == "sandbox-test"
        assert data["state"] == "ready"
        assert data["session_id"] is None


class TestContainerState:
    """Tests for ContainerState enum."""

    def test_container_states(self):
        """Test container state values."""
        assert ContainerState.CREATING.value == "creating"
        assert ContainerState.READY.value == "ready"
        assert ContainerState.ACQUIRED.value == "acquired"
        assert ContainerState.UNHEALTHY.value == "unhealthy"
        assert ContainerState.DESTROYING.value == "destroying"


class TestPoolState:
    """Tests for PoolState enum."""

    def test_pool_states(self):
        """Test pool state values."""
        assert PoolState.STOPPED.value == "stopped"
        assert PoolState.STARTING.value == "starting"
        assert PoolState.RUNNING.value == "running"
        assert PoolState.STOPPING.value == "stopping"
        assert PoolState.DRAINING.value == "draining"


class TestPoolStats:
    """Tests for PoolStats dataclass."""

    def test_stats_defaults(self):
        """Test default stats values."""
        stats = PoolStats()

        assert stats.total_containers == 0
        assert stats.ready_containers == 0
        assert stats.acquired_containers == 0
        assert stats.unhealthy_containers == 0
        assert stats.creating_containers == 0
        assert stats.total_acquisitions == 0
        assert stats.total_releases == 0
        assert stats.pool_utilization == 0.0

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = PoolStats(
            total_containers=10,
            ready_containers=5,
            acquired_containers=3,
        )
        data = stats.to_dict()

        assert data["total_containers"] == 10
        assert data["ready_containers"] == 5
        assert data["acquired_containers"] == 3


class TestContainerPool:
    """Tests for ContainerPool class."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return ContainerPoolConfig(
            min_pool_size=2,
            max_pool_size=5,
            warmup_count=2,
            health_check_interval_seconds=60.0,  # Slow down for tests
            creation_timeout_seconds=5.0,
        )

    @pytest.fixture
    def mock_pool(self, pool_config):
        """Create a pool with mocked Docker calls."""
        pool = ContainerPool(pool_config)
        return pool

    def test_pool_init(self, pool_config):
        """Test pool initialization."""
        pool = ContainerPool(pool_config)

        assert pool.config == pool_config
        assert pool.state == PoolState.STOPPED
        assert len(pool._containers) == 0

    def test_pool_default_config(self):
        """Test pool with default config."""
        pool = ContainerPool()

        assert pool.config is not None
        assert pool.config.min_pool_size == 5

    @pytest.mark.asyncio
    async def test_pool_start_already_running(self, mock_pool):
        """Test starting pool that's not stopped."""
        mock_pool._state = PoolState.RUNNING

        await mock_pool.start()  # Should log warning but not raise

        assert mock_pool.state == PoolState.RUNNING

    @pytest.mark.asyncio
    async def test_pool_stop_not_started(self, mock_pool):
        """Test stopping pool that's not started."""
        await mock_pool.stop()  # Should return early

        assert mock_pool.state == PoolState.STOPPED

    @pytest.mark.asyncio
    async def test_pool_acquire_not_running(self, mock_pool):
        """Test acquiring from non-running pool."""
        with pytest.raises(ContainerPoolError) as exc_info:
            await mock_pool.acquire("session-123")

        assert "Pool not running" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_release_missing_session(self, mock_pool):
        """Test releasing a session that doesn't exist."""
        mock_pool._state = PoolState.RUNNING

        # Should not raise, just log warning
        await mock_pool.release("nonexistent-session")

    @pytest.mark.asyncio
    async def test_get_container_for_session(self, mock_pool):
        """Test getting container for a session."""
        container = PooledContainer(
            container_id="abc123",
            container_name="sandbox-test",
        )
        mock_pool._containers["abc123"] = container
        mock_pool._session_containers["session-123"] = "abc123"

        result = mock_pool.get_container("session-123")

        assert result == container

    def test_get_container_unknown_session(self, mock_pool):
        """Test getting container for unknown session."""
        result = mock_pool.get_container("unknown-session")

        assert result is None

    def test_build_create_command(self, mock_pool):
        """Test building Docker create command."""
        cmd = mock_pool._build_create_command("test-container")

        assert "docker" in cmd
        assert "create" in cmd
        assert "--name" in cmd
        assert "test-container" in cmd
        assert f"--memory={mock_pool.config.memory_limit_mb}m" in cmd
        assert f"--cpus={mock_pool.config.cpu_limit}" in cmd
        assert "--security-opt=no-new-privileges" in cmd
        assert "--read-only" in cmd

    def test_record_acquire_time(self, mock_pool):
        """Test recording acquisition time."""
        mock_pool._record_acquire_time(50.0)
        mock_pool._record_acquire_time(100.0)

        assert len(mock_pool._acquire_times) == 2
        assert mock_pool._acquire_times[0] == 50.0

    def test_record_acquire_time_limit(self, mock_pool):
        """Test acquisition time list limited to 100."""
        for i in range(150):
            mock_pool._record_acquire_time(float(i))

        assert len(mock_pool._acquire_times) == 100
        # First items should have been popped
        assert mock_pool._acquire_times[0] == 50.0

    def test_record_creation_time(self, mock_pool):
        """Test recording creation time."""
        mock_pool._record_creation_time(1000.0)
        mock_pool._record_creation_time(1500.0)

        assert len(mock_pool._creation_times) == 2

    def test_get_stats(self, mock_pool):
        """Test getting pool statistics."""
        # Add some containers
        mock_pool._containers["c1"] = PooledContainer(
            container_id="c1",
            container_name="test-1",
            state=ContainerState.READY,
        )
        mock_pool._containers["c2"] = PooledContainer(
            container_id="c2",
            container_name="test-2",
            state=ContainerState.ACQUIRED,
        )
        mock_pool._containers["c3"] = PooledContainer(
            container_id="c3",
            container_name="test-3",
            state=ContainerState.UNHEALTHY,
        )

        mock_pool._acquire_times = [50.0, 100.0]
        mock_pool._creation_times = [1000.0, 1500.0]

        stats = mock_pool.stats

        assert stats.total_containers == 3
        assert stats.ready_containers == 1
        assert stats.acquired_containers == 1
        assert stats.unhealthy_containers == 1
        assert stats.avg_acquire_time_ms == 75.0
        assert stats.avg_creation_time_ms == 1250.0
        assert stats.pool_utilization == pytest.approx(1 / 3)


class TestContainerPoolWithMockedDocker:
    """Tests for ContainerPool with mocked Docker commands."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return ContainerPoolConfig(
            min_pool_size=1,
            max_pool_size=3,
            warmup_count=1,
            creation_timeout_seconds=2.0,
            acquire_timeout_seconds=1.0,
        )

    @pytest.mark.asyncio
    async def test_acquire_existing_session(self, pool_config):
        """Test acquiring container for existing session."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Pre-create a container bound to session
        container = PooledContainer(
            container_id="abc123",
            container_name="test-container",
            state=ContainerState.ACQUIRED,
            session_id="session-123",
        )
        pool._containers["abc123"] = container
        pool._session_containers["session-123"] = "abc123"

        result = await pool.acquire("session-123")

        assert result == container

    @pytest.mark.asyncio
    async def test_try_acquire_available(self, pool_config):
        """Test acquiring an available container."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        container = PooledContainer(
            container_id="abc123",
            container_name="test-container",
            state=ContainerState.READY,
        )
        pool._containers["abc123"] = container

        result = await pool._try_acquire("session-123")

        assert result is not None
        assert result.state == ContainerState.ACQUIRED
        assert result.session_id == "session-123"
        assert pool._session_containers["session-123"] == "abc123"

    @pytest.mark.asyncio
    async def test_try_acquire_none_available(self, pool_config):
        """Test acquiring when no containers available."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add acquired container
        container = PooledContainer(
            container_id="abc123",
            container_name="test-container",
            state=ContainerState.ACQUIRED,
            session_id="other-session",
        )
        pool._containers["abc123"] = container

        result = await pool._try_acquire("session-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_release_container(self, pool_config):
        """Test releasing a container."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        container = PooledContainer(
            container_id="abc123",
            container_name="test-container",
            state=ContainerState.ACQUIRED,
            session_id="session-123",
            execution_count=5,
        )
        pool._containers["abc123"] = container
        pool._session_containers["session-123"] = "abc123"

        await pool.release("session-123")

        assert container.state == ContainerState.READY
        assert container.session_id is None
        assert container.execution_count == 6  # Incremented
        assert "session-123" not in pool._session_containers

    @pytest.mark.asyncio
    async def test_destroy_session(self, pool_config):
        """Test destroying a session's container."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        container = PooledContainer(
            container_id="abc123",
            container_name="test-container",
            state=ContainerState.ACQUIRED,
            session_id="session-123",
        )
        pool._containers["abc123"] = container
        pool._session_containers["session-123"] = "abc123"

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock):
            await pool.destroy("session-123")

        assert "session-123" not in pool._session_containers


class TestContainerPoolScaling:
    """Tests for container pool scaling."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return ContainerPoolConfig(
            min_pool_size=2,
            max_pool_size=5,
            warmup_count=2,
        )

    @pytest.mark.asyncio
    async def test_scale_up(self, pool_config):
        """Test scaling up the pool."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        with patch.object(pool, "_create_container", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = PooledContainer(
                container_id="new-container",
                container_name="test-new",
            )

            created = await pool.scale_up(2)

            assert created == 2
            assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_scale_up_at_max(self, pool_config):
        """Test scaling up when at max capacity."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Fill pool to max
        for i in range(5):
            pool._containers[f"c{i}"] = PooledContainer(
                container_id=f"c{i}",
                container_name=f"test-{i}",
            )

        created = await pool.scale_up(3)

        assert created == 0

    @pytest.mark.asyncio
    async def test_scale_down(self, pool_config):
        """Test scaling down the pool."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add containers (need more than min_pool_size ready)
        for i in range(5):
            pool._containers[f"c{i}"] = PooledContainer(
                container_id=f"c{i}",
                container_name=f"test-{i}",
                state=ContainerState.READY,
                last_used_at=time.time() - i * 100,  # Varying idle times
            )

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock):
            removed = await pool.scale_down(2)

            # Should remove 2 (5 ready - 2 = 3, still > min_pool_size of 2)
            assert removed == 2

    @pytest.mark.asyncio
    async def test_scale_down_respects_minimum(self, pool_config):
        """Test scaling down respects minimum pool size."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add exactly min_pool_size containers
        for i in range(2):
            pool._containers[f"c{i}"] = PooledContainer(
                container_id=f"c{i}",
                container_name=f"test-{i}",
                state=ContainerState.READY,
            )

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock):
            removed = await pool.scale_down(5)

            # Should not remove any (at minimum)
            assert removed == 0

    @pytest.mark.asyncio
    async def test_scale_down_skips_acquired(self, pool_config):
        """Test scaling down skips acquired containers."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add mix of ready and acquired
        pool._containers["c1"] = PooledContainer(
            container_id="c1",
            container_name="test-1",
            state=ContainerState.ACQUIRED,
        )
        pool._containers["c2"] = PooledContainer(
            container_id="c2",
            container_name="test-2",
            state=ContainerState.READY,
        )
        pool._containers["c3"] = PooledContainer(
            container_id="c3",
            container_name="test-3",
            state=ContainerState.READY,
        )
        pool._containers["c4"] = PooledContainer(
            container_id="c4",
            container_name="test-4",
            state=ContainerState.READY,
        )

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock):
            removed = await pool.scale_down(5)

            # Only ready containers can be removed (3 - min_pool_size 2 = 1)
            assert removed == 1


class TestContainerPoolErrors:
    """Tests for container pool error handling."""

    def test_container_pool_error(self):
        """Test ContainerPoolError."""
        error = ContainerPoolError("Pool error")
        assert str(error) == "Pool error"

    def test_pool_exhausted_error(self):
        """Test PoolExhaustedError."""
        error = PoolExhaustedError("No containers available")
        assert str(error) == "No containers available"
        assert isinstance(error, ContainerPoolError)

    def test_container_creation_error(self):
        """Test ContainerCreationError."""
        error = ContainerCreationError("Creation failed")
        assert str(error) == "Creation failed"
        assert isinstance(error, ContainerPoolError)


class TestGlobalPoolInstance:
    """Tests for global pool instance management."""

    def test_get_container_pool_creates_instance(self):
        """Test getting global pool creates instance."""
        # Reset global
        set_container_pool(None)

        pool = get_container_pool()

        assert pool is not None
        assert isinstance(pool, ContainerPool)

    def test_get_container_pool_returns_same(self):
        """Test getting global pool returns same instance."""
        set_container_pool(None)

        pool1 = get_container_pool()
        pool2 = get_container_pool()

        assert pool1 is pool2

    def test_set_container_pool(self):
        """Test setting global pool."""
        custom_pool = ContainerPool(ContainerPoolConfig(min_pool_size=1))
        set_container_pool(custom_pool)

        pool = get_container_pool()

        assert pool is custom_pool

    def test_set_container_pool_none(self):
        """Test resetting global pool."""
        set_container_pool(None)

        # Getting pool should create new one
        pool = get_container_pool()
        assert pool is not None


class TestContainerPoolCleanup:
    """Tests for container pool cleanup functionality."""

    @pytest.fixture
    def pool_config(self):
        """Create a test pool configuration."""
        return ContainerPoolConfig(
            min_pool_size=2,
            max_pool_size=5,
            idle_timeout_seconds=100,
            max_container_age_seconds=1000,
        )

    @pytest.mark.asyncio
    async def test_cleanup_unhealthy_containers(self, pool_config):
        """Test cleanup removes unhealthy containers."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add healthy and unhealthy containers
        pool._containers["c1"] = PooledContainer(
            container_id="c1",
            container_name="test-1",
            state=ContainerState.READY,
        )
        pool._containers["c2"] = PooledContainer(
            container_id="c2",
            container_name="test-2",
            state=ContainerState.UNHEALTHY,
        )

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock) as mock_destroy:
            await pool._cleanup_expired_containers()

            # Should destroy unhealthy container
            mock_destroy.assert_called_once_with("c2")

    @pytest.mark.asyncio
    async def test_cleanup_expired_idle_containers(self, pool_config):
        """Test cleanup removes idle containers."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add containers - some idle, some recent
        pool._containers["c1"] = PooledContainer(
            container_id="c1",
            container_name="test-1",
            state=ContainerState.READY,
            last_used_at=time.time() - 200,  # Very idle
        )
        pool._containers["c2"] = PooledContainer(
            container_id="c2",
            container_name="test-2",
            state=ContainerState.READY,
            last_used_at=time.time(),  # Just used
        )
        pool._containers["c3"] = PooledContainer(
            container_id="c3",
            container_name="test-3",
            state=ContainerState.READY,
            last_used_at=time.time(),  # Just used
        )

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock) as mock_destroy:
            await pool._cleanup_expired_containers()

            # c1 is expired (idle > 100s) and can be removed (3 ready > min 2)
            mock_destroy.assert_called_once_with("c1")

    @pytest.mark.asyncio
    async def test_cleanup_respects_minimum(self, pool_config):
        """Test cleanup respects minimum pool size."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add only min_pool_size containers, all idle
        pool._containers["c1"] = PooledContainer(
            container_id="c1",
            container_name="test-1",
            state=ContainerState.READY,
            last_used_at=time.time() - 200,  # Idle
        )
        pool._containers["c2"] = PooledContainer(
            container_id="c2",
            container_name="test-2",
            state=ContainerState.READY,
            last_used_at=time.time() - 200,  # Idle
        )

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock) as mock_destroy:
            await pool._cleanup_expired_containers()

            # Should not remove any (at minimum)
            mock_destroy.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_skips_acquired(self, pool_config):
        """Test cleanup skips acquired containers."""
        pool = ContainerPool(pool_config)
        pool._state = PoolState.RUNNING

        # Add acquired container that appears expired
        pool._containers["c1"] = PooledContainer(
            container_id="c1",
            container_name="test-1",
            state=ContainerState.ACQUIRED,
            last_used_at=time.time() - 200,  # Would be idle, but acquired
        )

        with patch.object(pool, "_destroy_container", new_callable=AsyncMock) as mock_destroy:
            await pool._cleanup_expired_containers()

            mock_destroy.assert_not_called()
