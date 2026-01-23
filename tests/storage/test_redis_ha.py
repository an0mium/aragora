"""
Tests for Redis High-Availability Connection Factory.

Tests cover:
- Configuration parsing from environment variables
- Client creation for all three modes (standalone, sentinel, cluster)
- Health check functionality
- Singleton/caching behavior
- Error handling and fallback scenarios
"""

from __future__ import annotations

import os
from typing import Any, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.storage.redis_ha import (
    RedisHAConfig,
    RedisMode,
    _parse_host_port,
    _parse_redis_url,
    check_redis_health,
    get_async_redis_client,
    get_cached_redis_client,
    get_redis_client,
    reset_cached_clients,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Remove Redis-related environment variables for clean tests."""
    env_vars = [
        "ARAGORA_REDIS_MODE",
        "ARAGORA_REDIS_URL",
        "REDIS_URL",
        "ARAGORA_REDIS_HOST",
        "ARAGORA_REDIS_PORT",
        "ARAGORA_REDIS_PASSWORD",
        "ARAGORA_REDIS_DB",
        "ARAGORA_REDIS_SENTINEL_HOSTS",
        "ARAGORA_REDIS_SENTINEL_MASTER",
        "ARAGORA_REDIS_SENTINEL_PASSWORD",
        "ARAGORA_REDIS_CLUSTER_NODES",
        "ARAGORA_REDIS_CLUSTER_READ_FROM_REPLICAS",
        "ARAGORA_REDIS_SOCKET_TIMEOUT",
        "ARAGORA_REDIS_MAX_CONNECTIONS",
        "ARAGORA_REDIS_SSL",
    ]
    original = {k: os.environ.get(k) for k in env_vars}

    # Clear all Redis env vars
    for var in env_vars:
        os.environ.pop(var, None)

    yield

    # Restore original values
    for var, value in original.items():
        if value is not None:
            os.environ[var] = value
        else:
            os.environ.pop(var, None)


@pytest.fixture
def mock_redis() -> Generator[MagicMock, None, None]:
    """Mock the redis module."""
    with patch("aragora.storage.redis_ha.redis") as mock:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.info.return_value = {
            "redis_version": "7.0.0",
            "connected_clients": 10,
            "used_memory_human": "1.5M",
            "role": "master",
        }

        mock_pool = MagicMock()
        mock.ConnectionPool.return_value = mock_pool
        mock.ConnectionPool.from_url.return_value = mock_pool
        mock.Redis.return_value = mock_client

        yield mock


@pytest.fixture(autouse=True)
def reset_clients() -> Generator[None, None, None]:
    """Reset cached clients before and after each test."""
    reset_cached_clients()
    yield
    reset_cached_clients()


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRedisHAConfig:
    """Tests for RedisHAConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RedisHAConfig()

        assert config.mode == RedisMode.STANDALONE
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.socket_timeout == 5.0
        assert config.max_connections == 50
        assert config.decode_responses is True
        assert config.sentinel_hosts == []
        assert config.cluster_nodes == []

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RedisHAConfig(
            mode=RedisMode.SENTINEL,
            host="redis.example.com",
            port=6380,
            password="secret",
            db=1,
            sentinel_hosts=["sentinel1:26379", "sentinel2:26379"],
            sentinel_master="custom-master",
            socket_timeout=10.0,
            ssl=True,
        )

        assert config.mode == RedisMode.SENTINEL
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.db == 1
        assert len(config.sentinel_hosts) == 2
        assert config.sentinel_master == "custom-master"
        assert config.socket_timeout == 10.0
        assert config.ssl is True

    def test_from_env_standalone(self, clean_env: None) -> None:
        """Test configuration from environment for standalone mode."""
        os.environ["ARAGORA_REDIS_MODE"] = "standalone"
        os.environ["ARAGORA_REDIS_HOST"] = "redis.example.com"
        os.environ["ARAGORA_REDIS_PORT"] = "6380"
        os.environ["ARAGORA_REDIS_PASSWORD"] = "secret"
        os.environ["ARAGORA_REDIS_DB"] = "2"

        config = RedisHAConfig.from_env()

        assert config.mode == RedisMode.STANDALONE
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.db == 2

    def test_from_env_sentinel(self, clean_env: None) -> None:
        """Test configuration from environment for sentinel mode."""
        os.environ["ARAGORA_REDIS_MODE"] = "sentinel"
        os.environ["ARAGORA_REDIS_SENTINEL_HOSTS"] = (
            "sentinel1:26379,sentinel2:26379,sentinel3:26379"
        )
        os.environ["ARAGORA_REDIS_SENTINEL_MASTER"] = "mymaster"
        os.environ["ARAGORA_REDIS_SENTINEL_PASSWORD"] = "sentinel-secret"
        os.environ["ARAGORA_REDIS_PASSWORD"] = "redis-secret"

        config = RedisHAConfig.from_env()

        assert config.mode == RedisMode.SENTINEL
        assert len(config.sentinel_hosts) == 3
        assert "sentinel1:26379" in config.sentinel_hosts
        assert config.sentinel_master == "mymaster"
        assert config.sentinel_password == "sentinel-secret"
        assert config.password == "redis-secret"

    def test_from_env_cluster(self, clean_env: None) -> None:
        """Test configuration from environment for cluster mode."""
        os.environ["ARAGORA_REDIS_MODE"] = "cluster"
        os.environ["ARAGORA_REDIS_CLUSTER_NODES"] = "node1:6379,node2:6379,node3:6379"
        os.environ["ARAGORA_REDIS_CLUSTER_READ_FROM_REPLICAS"] = "false"
        os.environ["ARAGORA_REDIS_PASSWORD"] = "cluster-secret"

        config = RedisHAConfig.from_env()

        assert config.mode == RedisMode.CLUSTER
        assert len(config.cluster_nodes) == 3
        assert "node1:6379" in config.cluster_nodes
        assert config.cluster_read_from_replicas is False
        assert config.password == "cluster-secret"

    def test_from_env_auto_detect_sentinel(self, clean_env: None) -> None:
        """Test auto-detection of sentinel mode from environment."""
        # Don't set mode explicitly, but provide sentinel hosts
        os.environ["ARAGORA_REDIS_SENTINEL_HOSTS"] = "sentinel1:26379,sentinel2:26379"

        config = RedisHAConfig.from_env()

        assert config.mode == RedisMode.SENTINEL
        assert len(config.sentinel_hosts) == 2

    def test_from_env_auto_detect_cluster(self, clean_env: None) -> None:
        """Test auto-detection of cluster mode from environment."""
        # Don't set mode explicitly, but provide cluster nodes
        os.environ["ARAGORA_REDIS_CLUSTER_NODES"] = "node1:6379,node2:6379"

        config = RedisHAConfig.from_env()

        assert config.mode == RedisMode.CLUSTER
        assert len(config.cluster_nodes) == 2

    def test_from_env_url_parsing(self, clean_env: None) -> None:
        """Test URL parsing from environment."""
        os.environ["ARAGORA_REDIS_URL"] = "redis://user:pass@redis.example.com:6380/3"

        config = RedisHAConfig.from_env()

        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.url == "redis://user:pass@redis.example.com:6380/3"

    def test_from_env_invalid_mode(self, clean_env: None) -> None:
        """Test fallback for invalid mode."""
        os.environ["ARAGORA_REDIS_MODE"] = "invalid_mode"

        config = RedisHAConfig.from_env()

        # Should fall back to standalone
        assert config.mode == RedisMode.STANDALONE

    def test_get_parsed_sentinel_hosts(self) -> None:
        """Test parsing sentinel hosts."""
        config = RedisHAConfig(sentinel_hosts=["sentinel1:26379", "sentinel2:26380", "sentinel3"])

        parsed = config.get_parsed_sentinel_hosts()

        assert len(parsed) == 3
        assert parsed[0] == ("sentinel1", 26379)
        assert parsed[1] == ("sentinel2", 26380)
        assert parsed[2] == ("sentinel3", 26379)  # Default port

    def test_get_parsed_cluster_nodes(self) -> None:
        """Test parsing cluster nodes."""
        config = RedisHAConfig(cluster_nodes=["node1:6379", "node2:6380", "node3"])

        parsed = config.get_parsed_cluster_nodes()

        assert len(parsed) == 3
        assert parsed[0] == ("node1", 6379)
        assert parsed[1] == ("node2", 6380)
        assert parsed[2] == ("node3", 6379)  # Default port


class TestParseHelpers:
    """Tests for URL and host parsing helpers."""

    def test_parse_host_port_with_port(self) -> None:
        """Test parsing host:port string."""
        host, port = _parse_host_port("redis.example.com:6380")

        assert host == "redis.example.com"
        assert port == 6380

    def test_parse_host_port_without_port(self) -> None:
        """Test parsing host without port."""
        host, port = _parse_host_port("redis.example.com")

        assert host == "redis.example.com"
        assert port == 6379  # Default

    def test_parse_host_port_custom_default(self) -> None:
        """Test parsing with custom default port."""
        host, port = _parse_host_port("sentinel.example.com", default_port=26379)

        assert host == "sentinel.example.com"
        assert port == 26379

    def test_parse_host_port_invalid_port(self) -> None:
        """Test parsing with invalid port."""
        host, port = _parse_host_port("redis.example.com:invalid")

        assert host == "redis.example.com"
        assert port == 6379  # Falls back to default

    def test_parse_redis_url_full(self) -> None:
        """Test parsing full Redis URL."""
        host, port = _parse_redis_url("redis://user:pass@redis.example.com:6380/0")

        assert host == "redis.example.com"
        assert port == 6380

    def test_parse_redis_url_simple(self) -> None:
        """Test parsing simple Redis URL."""
        host, port = _parse_redis_url("redis://localhost:6379")

        assert host == "localhost"
        assert port == 6379

    def test_parse_redis_url_no_port(self) -> None:
        """Test parsing URL without port."""
        host, port = _parse_redis_url("redis://redis.example.com")

        assert host == "redis.example.com"
        assert port == 6379


# =============================================================================
# Client Creation Tests
# =============================================================================


class TestGetRedisClient:
    """Tests for get_redis_client function."""

    def test_standalone_client_creation(self, clean_env: None) -> None:
        """Test standalone client creation."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch(
            "aragora.storage.redis_ha._create_standalone_client",
            return_value=mock_client,
        ) as mock_create:
            config = RedisHAConfig(mode=RedisMode.STANDALONE, host="localhost", port=6379)
            client = get_redis_client(config)

            assert client is not None
            mock_create.assert_called_once_with(config)

    def test_standalone_client_with_url(self, clean_env: None) -> None:
        """Test standalone client creation with URL."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch(
            "aragora.storage.redis_ha._create_standalone_client",
            return_value=mock_client,
        ) as mock_create:
            config = RedisHAConfig(
                mode=RedisMode.STANDALONE,
                url="redis://localhost:6379/0",
            )
            client = get_redis_client(config)

            assert client is not None
            mock_create.assert_called_once_with(config)

    def test_sentinel_client_creation(self, clean_env: None) -> None:
        """Test sentinel client creation."""
        mock_master = MagicMock()
        mock_master.ping.return_value = True

        with patch(
            "aragora.storage.redis_ha._create_sentinel_client",
            return_value=mock_master,
        ) as mock_create:
            config = RedisHAConfig(
                mode=RedisMode.SENTINEL,
                sentinel_hosts=["sentinel1:26379", "sentinel2:26379"],
                sentinel_master="mymaster",
            )
            client = get_redis_client(config)

            assert client is not None
            mock_create.assert_called_once_with(config)

    def test_cluster_client_creation(self, clean_env: None) -> None:
        """Test cluster client creation."""
        mock_cluster_client = MagicMock()
        mock_cluster_client.ping.return_value = True

        with patch(
            "aragora.storage.redis_ha._create_cluster_client",
            return_value=mock_cluster_client,
        ) as mock_create:
            config = RedisHAConfig(
                mode=RedisMode.CLUSTER,
                cluster_nodes=["node1:6379", "node2:6379"],
            )
            client = get_redis_client(config)

            assert client is not None
            mock_create.assert_called_once_with(config)

    def test_client_creation_import_error(self, clean_env: None) -> None:
        """Test graceful handling of missing redis package."""
        with patch(
            "aragora.storage.redis_ha._create_standalone_client",
            side_effect=ImportError("No module named 'redis'"),
        ):
            config = RedisHAConfig(mode=RedisMode.STANDALONE)
            client = get_redis_client(config)

            assert client is None

    def test_client_creation_connection_error(self, clean_env: None) -> None:
        """Test graceful handling of connection errors."""
        with patch(
            "aragora.storage.redis_ha._create_standalone_client",
            side_effect=ConnectionError("Connection refused"),
        ):
            config = RedisHAConfig(mode=RedisMode.STANDALONE)
            client = get_redis_client(config)

            assert client is None


class TestGetAsyncRedisClient:
    """Tests for get_async_redis_client function."""

    @pytest.mark.asyncio
    async def test_async_standalone_client_creation(self, clean_env: None) -> None:
        """Test async standalone client creation."""
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)

        with patch(
            "aragora.storage.redis_ha._create_async_standalone_client",
            return_value=mock_client,
        ):
            config = RedisHAConfig(mode=RedisMode.STANDALONE)
            client = await get_async_redis_client(config)

            assert client is not None

    @pytest.mark.asyncio
    async def test_async_client_creation_error(self, clean_env: None) -> None:
        """Test async client creation error handling."""
        with patch(
            "aragora.storage.redis_ha._create_async_standalone_client",
            side_effect=ImportError("No module named 'redis'"),
        ):
            config = RedisHAConfig(mode=RedisMode.STANDALONE)
            client = await get_async_redis_client(config)

            assert client is None


# =============================================================================
# Caching/Singleton Tests
# =============================================================================


class TestCachedClients:
    """Tests for cached client management."""

    def test_get_cached_client_returns_same_instance(self, clean_env: None) -> None:
        """Test that cached client returns same instance."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            return mock_client

        with patch(
            "aragora.storage.redis_ha.get_redis_client",
            side_effect=mock_create,
        ):
            config = RedisHAConfig(mode=RedisMode.STANDALONE)

            client1 = get_cached_redis_client(config)
            client2 = get_cached_redis_client(config)

            # Should be the same instance (cached)
            assert client1 is client2
            # Should only create once
            assert call_count[0] == 1

    def test_reset_cached_clients(self, clean_env: None) -> None:
        """Test resetting cached clients."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        call_count = [0]

        def mock_create(*args, **kwargs):
            call_count[0] += 1
            return mock_client

        with patch(
            "aragora.storage.redis_ha.get_redis_client",
            side_effect=mock_create,
        ):
            config = RedisHAConfig(mode=RedisMode.STANDALONE)

            client1 = get_cached_redis_client(config)
            reset_cached_clients()
            client2 = get_cached_redis_client(config)

            # Should create twice after reset
            assert call_count[0] == 2
            mock_client.close.assert_called_once()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_healthy(self, clean_env: None) -> None:
        """Test health check returns healthy status."""
        with patch("aragora.storage.redis_ha.get_redis_client") as mock_get:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.info.return_value = {
                "redis_version": "7.0.0",
                "connected_clients": 10,
                "used_memory_human": "1.5M",
                "role": "master",
            }
            mock_get.return_value = mock_client

            result = check_redis_health()

            assert result["healthy"] is True
            assert result["mode"] == "standalone"
            assert result["error"] is None
            assert result["latency_ms"] is not None
            assert result["info"]["redis_version"] == "7.0.0"

    def test_health_check_unhealthy(self, clean_env: None) -> None:
        """Test health check returns unhealthy status on error."""
        with patch("aragora.storage.redis_ha.get_redis_client") as mock_get:
            mock_get.return_value = None

            result = check_redis_health()

            assert result["healthy"] is False
            assert result["error"] == "Failed to create client"

    def test_health_check_connection_error(self, clean_env: None) -> None:
        """Test health check handles connection errors."""
        with patch("aragora.storage.redis_ha.get_redis_client") as mock_get:
            mock_client = MagicMock()
            mock_client.ping.side_effect = ConnectionError("Connection refused")
            mock_get.return_value = mock_client

            result = check_redis_health()

            assert result["healthy"] is False
            assert "Connection refused" in result["error"]


# =============================================================================
# SSL/TLS Tests
# =============================================================================


class TestSSLConfiguration:
    """Tests for SSL/TLS configuration."""

    def test_ssl_config_from_env(self, clean_env: None) -> None:
        """Test SSL configuration from environment."""
        os.environ["ARAGORA_REDIS_SSL"] = "true"
        os.environ["ARAGORA_REDIS_SSL_CERT_REQS"] = "required"
        os.environ["ARAGORA_REDIS_SSL_CA_CERTS"] = "/path/to/ca.crt"

        config = RedisHAConfig.from_env()

        assert config.ssl is True
        assert config.ssl_cert_reqs == "required"
        assert config.ssl_ca_certs == "/path/to/ca.crt"

    def test_ssl_disabled_by_default(self, clean_env: None) -> None:
        """Test SSL is disabled by default."""
        config = RedisHAConfig.from_env()

        assert config.ssl is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_empty_sentinel_hosts(self, clean_env: None) -> None:
        """Test error when no sentinel hosts configured."""
        config = RedisHAConfig(mode=RedisMode.SENTINEL, sentinel_hosts=[])

        # This should be handled gracefully
        with patch(
            "aragora.storage.redis_ha._create_sentinel_client",
            side_effect=ValueError("No sentinel hosts configured"),
        ):
            client = get_redis_client(config)
            assert client is None

    def test_empty_cluster_nodes(self, clean_env: None) -> None:
        """Test error when no cluster nodes configured."""
        config = RedisHAConfig(mode=RedisMode.CLUSTER, cluster_nodes=[])

        with patch(
            "aragora.storage.redis_ha._create_cluster_client",
            side_effect=ValueError("No cluster nodes configured"),
        ):
            client = get_redis_client(config)
            assert client is None


# =============================================================================
# Integration Tests (require actual Redis)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("REDIS_INTEGRATION_TEST") != "true",
    reason="Redis integration tests disabled",
)
class TestRedisIntegration:
    """Integration tests requiring actual Redis server."""

    def test_standalone_integration(self) -> None:
        """Test actual connection to standalone Redis."""
        config = RedisHAConfig(mode=RedisMode.STANDALONE, host="localhost", port=6379)
        client = get_redis_client(config)

        if client is None:
            pytest.skip("Redis not available")

        # Test basic operations
        client.set("test_key", "test_value")
        value = client.get("test_key")
        assert value == "test_value"

        # Cleanup
        client.delete("test_key")

    @pytest.mark.asyncio
    async def test_async_standalone_integration(self) -> None:
        """Test actual async connection to standalone Redis."""
        config = RedisHAConfig(mode=RedisMode.STANDALONE, host="localhost", port=6379)
        client = await get_async_redis_client(config)

        if client is None:
            pytest.skip("Redis not available")

        # Test basic operations
        await client.set("test_async_key", "test_async_value")
        value = await client.get("test_async_key")
        assert value == "test_async_value"

        # Cleanup
        await client.delete("test_async_key")
        await client.close()
