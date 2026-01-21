"""Tests for Redis state backend initialization during server startup.

Verifies that Redis state management is properly initialized when
configured for horizontal scaling.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRedisStateStartup:
    """Test Redis state backend initialization."""

    @pytest.mark.asyncio
    async def test_init_redis_disabled_by_default(self):
        """Redis backend should be disabled when not configured."""
        from aragora.server.startup import init_redis_state_backend

        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing redis env vars
            env_without_redis = {
                k: v for k, v in os.environ.items()
                if "REDIS" not in k and "STATE_BACKEND" not in k
            }
            with patch.dict(os.environ, env_without_redis, clear=True):
                result = await init_redis_state_backend()

        assert result is False

    @pytest.mark.asyncio
    async def test_init_redis_enabled_with_state_backend(self):
        """Redis backend should be enabled when ARAGORA_STATE_BACKEND=redis."""
        from aragora.server.startup import init_redis_state_backend

        mock_manager = MagicMock()
        mock_manager.is_connected = True

        with patch.dict(os.environ, {"ARAGORA_STATE_BACKEND": "redis"}):
            with patch(
                "aragora.server.redis_state.get_redis_state_manager",
                new_callable=AsyncMock,
                return_value=mock_manager,
            ):
                result = await init_redis_state_backend()

        assert result is True

    @pytest.mark.asyncio
    async def test_init_redis_enabled_with_redis_url(self):
        """Redis backend should be enabled when ARAGORA_REDIS_URL is set."""
        from aragora.server.startup import init_redis_state_backend

        mock_manager = MagicMock()
        mock_manager.is_connected = True

        with patch.dict(os.environ, {"ARAGORA_REDIS_URL": "redis://localhost:6379"}):
            with patch(
                "aragora.server.redis_state.get_redis_state_manager",
                new_callable=AsyncMock,
                return_value=mock_manager,
            ):
                result = await init_redis_state_backend()

        assert result is True

    @pytest.mark.asyncio
    async def test_init_redis_returns_false_on_connection_failure(self):
        """Should return False when Redis connection fails."""
        from aragora.server.startup import init_redis_state_backend

        mock_manager = MagicMock()
        mock_manager.is_connected = False

        with patch.dict(os.environ, {"ARAGORA_STATE_BACKEND": "redis"}):
            with patch(
                "aragora.server.redis_state.get_redis_state_manager",
                new_callable=AsyncMock,
                return_value=mock_manager,
            ):
                result = await init_redis_state_backend()

        assert result is False

    @pytest.mark.asyncio
    async def test_init_redis_handles_import_error(self):
        """Should return False when redis_state module not available."""
        from aragora.server.startup import init_redis_state_backend

        with patch.dict(os.environ, {"ARAGORA_STATE_BACKEND": "redis"}):
            with patch(
                "aragora.server.redis_state.get_redis_state_manager",
                new_callable=AsyncMock,
                side_effect=ImportError("aioredis not installed"),
            ):
                result = await init_redis_state_backend()

        assert result is False

    @pytest.mark.asyncio
    async def test_init_redis_handles_runtime_error(self):
        """Should return False on runtime errors without crashing."""
        from aragora.server.startup import init_redis_state_backend

        with patch.dict(os.environ, {"ARAGORA_STATE_BACKEND": "redis"}):
            with patch(
                "aragora.server.redis_state.get_redis_state_manager",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Connection refused"),
            ):
                result = await init_redis_state_backend()

        assert result is False


class TestRedisStateManager:
    """Test RedisStateManager functionality if available."""

    def test_redis_state_module_exists(self):
        """Redis state module should be importable."""
        try:
            from aragora.server import redis_state
            assert hasattr(redis_state, "RedisStateManager")
            assert hasattr(redis_state, "get_redis_state_manager")
        except ImportError:
            pytest.skip("Redis state module not available")

    def test_redis_state_manager_has_required_methods(self):
        """RedisStateManager should have required interface methods."""
        try:
            from aragora.server.redis_state import RedisStateManager

            # Check required methods exist
            assert hasattr(RedisStateManager, "connect")
            assert hasattr(RedisStateManager, "disconnect")
            assert hasattr(RedisStateManager, "is_connected")
        except ImportError:
            pytest.skip("Redis state module not available")
