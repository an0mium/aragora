"""
Tests for aragora.server.startup.redis module.

Tests Redis HA and Redis state backend initialization.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# init_redis_ha Tests
# =============================================================================


class TestInitRedisHA:
    """Tests for init_redis_ha function."""

    @pytest.mark.asyncio
    async def test_not_configured(self) -> None:
        """Test Redis HA not configured returns disabled status."""
        mock_config = MagicMock()
        mock_config.mode = MagicMock(value="standalone")
        mock_config.enabled = False
        mock_config.is_configured = False

        mock_redis_config = MagicMock()
        mock_redis_config.get_redis_ha_config = MagicMock(return_value=mock_config)

        mock_redis_ha = MagicMock()
        mock_redis_ha.RedisHAConfig = MagicMock
        mock_redis_ha.RedisMode = MagicMock
        mock_redis_ha.check_redis_health = MagicMock()
        mock_redis_ha.get_redis_client = MagicMock()
        mock_redis_ha.reset_cached_clients = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.config.redis": mock_redis_config,
                "aragora.storage.redis_ha": mock_redis_ha,
            },
        ):
            from aragora.server.startup.redis import init_redis_ha

            result = await init_redis_ha()

        assert result["enabled"] is False
        assert result["mode"] == "standalone"
        assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_healthy_connection(self) -> None:
        """Test successful Redis HA connection."""
        mock_config = MagicMock()
        mock_config.mode = MagicMock(value="sentinel")
        mock_config.enabled = True
        mock_config.is_configured = True
        mock_config.host = "localhost"
        mock_config.port = 6379
        mock_config.password = None
        mock_config.db = 0
        mock_config.url = "redis://localhost:6379"
        mock_config.sentinel_hosts = [("localhost", 26379)]
        mock_config.sentinel_master = "mymaster"
        mock_config.sentinel_password = None
        mock_config.cluster_nodes = []
        mock_config.cluster_read_from_replicas = False
        mock_config.cluster_skip_full_coverage_check = False
        mock_config.socket_timeout = 5.0
        mock_config.socket_connect_timeout = 2.0
        mock_config.max_connections = 50
        mock_config.retry_on_timeout = True
        mock_config.health_check_interval = 30
        mock_config.decode_responses = True
        mock_config.ssl = False
        mock_config.ssl_cert_reqs = "required"
        mock_config.ssl_ca_certs = None
        mock_config.get_mode_description = MagicMock(return_value="Redis Sentinel HA")

        mock_redis_config = MagicMock()
        mock_redis_config.get_redis_ha_config = MagicMock(return_value=mock_config)

        mock_client = MagicMock()
        mock_redis_ha = MagicMock()
        mock_redis_ha.RedisHAConfig = MagicMock()
        mock_redis_ha.RedisMode = MagicMock(return_value="sentinel")
        mock_redis_ha.check_redis_health = MagicMock(
            return_value={"healthy": True, "latency_ms": 1.5}
        )
        mock_redis_ha.get_redis_client = MagicMock(return_value=mock_client)
        mock_redis_ha.reset_cached_clients = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.config.redis": mock_redis_config,
                "aragora.storage.redis_ha": mock_redis_ha,
            },
        ):
            from aragora.server.startup.redis import init_redis_ha

            result = await init_redis_ha()

        assert result["enabled"] is True
        assert result["mode"] == "sentinel"
        assert result["healthy"] is True
        assert result["description"] == "Redis Sentinel HA"

    @pytest.mark.asyncio
    async def test_health_check_failed(self) -> None:
        """Test Redis HA health check failure."""
        mock_config = MagicMock()
        mock_config.mode = MagicMock(value="standalone")
        mock_config.enabled = True
        mock_config.is_configured = True
        mock_config.host = "localhost"
        mock_config.port = 6379
        mock_config.password = None
        mock_config.db = 0
        mock_config.url = "redis://localhost:6379"
        mock_config.sentinel_hosts = []
        mock_config.sentinel_master = "mymaster"
        mock_config.sentinel_password = None
        mock_config.cluster_nodes = []
        mock_config.cluster_read_from_replicas = False
        mock_config.cluster_skip_full_coverage_check = False
        mock_config.socket_timeout = 5.0
        mock_config.socket_connect_timeout = 2.0
        mock_config.max_connections = 50
        mock_config.retry_on_timeout = True
        mock_config.health_check_interval = 30
        mock_config.decode_responses = True
        mock_config.ssl = False
        mock_config.ssl_cert_reqs = "required"
        mock_config.ssl_ca_certs = None
        mock_config.get_mode_description = MagicMock(return_value="Redis standalone")

        mock_redis_config = MagicMock()
        mock_redis_config.get_redis_ha_config = MagicMock(return_value=mock_config)

        mock_redis_ha = MagicMock()
        mock_redis_ha.RedisHAConfig = MagicMock()
        mock_redis_ha.RedisMode = MagicMock(return_value="standalone")
        mock_redis_ha.check_redis_health = MagicMock(
            return_value={"healthy": False, "error": "Connection refused"}
        )
        mock_redis_ha.reset_cached_clients = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.config.redis": mock_redis_config,
                "aragora.storage.redis_ha": mock_redis_ha,
            },
        ):
            from aragora.server.startup.redis import init_redis_ha

            result = await init_redis_ha()

        assert result["healthy"] is False
        assert result["error"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_import_error(self) -> None:
        """Test ImportError handling."""
        with patch.dict(
            "sys.modules",
            {
                "aragora.config.redis": None,
                "aragora.storage.redis_ha": None,
            },
        ):
            import importlib
            import aragora.server.startup.redis as redis_module

            importlib.reload(redis_module)
            result = await redis_module.init_redis_ha()

        assert result["enabled"] is False
        assert "not installed" in result["error"]


# =============================================================================
# init_redis_state_backend Tests
# =============================================================================


class TestInitRedisStateBackend:
    """Tests for init_redis_state_backend function."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Redis state backend disabled by default."""
        monkeypatch.delenv("ARAGORA_STATE_BACKEND", raising=False)
        monkeypatch.delenv("ARAGORA_REDIS_URL", raising=False)

        from aragora.server.startup.redis import init_redis_state_backend

        result = await init_redis_state_backend()
        assert result is False

    @pytest.mark.asyncio
    async def test_enabled_with_backend_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Redis state backend enabled via ARAGORA_STATE_BACKEND."""
        monkeypatch.setenv("ARAGORA_STATE_BACKEND", "redis")

        mock_manager = MagicMock()
        mock_manager.is_connected = True

        mock_redis_state = MagicMock()
        mock_redis_state.get_redis_state_manager = AsyncMock(return_value=mock_manager)

        with patch.dict("sys.modules", {"aragora.server.redis_state": mock_redis_state}):
            from aragora.server.startup.redis import init_redis_state_backend

            result = await init_redis_state_backend()

        assert result is True
        mock_redis_state.get_redis_state_manager.assert_awaited_once_with(auto_connect=True)

    @pytest.mark.asyncio
    async def test_enabled_with_redis_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Redis state backend enabled via ARAGORA_REDIS_URL."""
        monkeypatch.delenv("ARAGORA_STATE_BACKEND", raising=False)
        monkeypatch.setenv("ARAGORA_REDIS_URL", "redis://localhost:6379")

        mock_manager = MagicMock()
        mock_manager.is_connected = True

        mock_redis_state = MagicMock()
        mock_redis_state.get_redis_state_manager = AsyncMock(return_value=mock_manager)

        with patch.dict("sys.modules", {"aragora.server.redis_state": mock_redis_state}):
            from aragora.server.startup.redis import init_redis_state_backend

            result = await init_redis_state_backend()

        assert result is True

    @pytest.mark.asyncio
    async def test_connection_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Redis state backend connection failure."""
        monkeypatch.setenv("ARAGORA_STATE_BACKEND", "redis")

        mock_manager = MagicMock()
        mock_manager.is_connected = False

        mock_redis_state = MagicMock()
        mock_redis_state.get_redis_state_manager = AsyncMock(return_value=mock_manager)

        with patch.dict("sys.modules", {"aragora.server.redis_state": mock_redis_state}):
            from aragora.server.startup.redis import init_redis_state_backend

            result = await init_redis_state_backend()

        assert result is False

    @pytest.mark.asyncio
    async def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ImportError handling."""
        monkeypatch.setenv("ARAGORA_STATE_BACKEND", "redis")

        with patch.dict("sys.modules", {"aragora.server.redis_state": None}):
            import importlib
            import aragora.server.startup.redis as redis_module

            importlib.reload(redis_module)
            result = await redis_module.init_redis_state_backend()

        assert result is False

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ConnectionError handling."""
        monkeypatch.setenv("ARAGORA_STATE_BACKEND", "redis")

        mock_redis_state = MagicMock()
        mock_redis_state.get_redis_state_manager = AsyncMock(
            side_effect=ConnectionError("Redis unreachable")
        )

        with patch.dict("sys.modules", {"aragora.server.redis_state": mock_redis_state}):
            from aragora.server.startup.redis import init_redis_state_backend

            result = await init_redis_state_backend()

        assert result is False
