"""
Tests for aragora.server.startup.database module.

Tests PostgreSQL pool initialization and shutdown.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# init_postgres_pool Tests
# =============================================================================


class TestInitPostgresPool:
    """Tests for init_postgres_pool function."""

    @pytest.mark.asyncio
    async def test_disabled_via_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test shared pool disabled via environment variable."""
        monkeypatch.setenv("ARAGORA_USE_SHARED_POOL", "false")

        from aragora.server.startup.database import init_postgres_pool

        result = await init_postgres_pool()

        assert result["enabled"] is False
        assert result["backend"] == "sqlite"
        assert result["reason"] == "disabled_by_env"

    @pytest.mark.asyncio
    async def test_sqlite_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test falls back to SQLite when PostgreSQL not configured."""
        monkeypatch.setenv("ARAGORA_USE_SHARED_POOL", "true")

        mock_factory = MagicMock()
        mock_factory.StorageBackend = MagicMock()
        mock_factory.StorageBackend.SQLITE = "sqlite"
        mock_factory.get_storage_backend = MagicMock(return_value="sqlite")

        with patch.dict("sys.modules", {"aragora.storage.factory": mock_factory}):
            from aragora.server.startup.database import init_postgres_pool

            result = await init_postgres_pool()

        assert result["enabled"] is False
        assert result["backend"] == "sqlite"

    @pytest.mark.asyncio
    async def test_successful_pool_initialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful PostgreSQL pool initialization."""
        monkeypatch.setenv("ARAGORA_USE_SHARED_POOL", "true")
        monkeypatch.setenv("ARAGORA_POOL_MIN_SIZE", "3")
        monkeypatch.setenv("ARAGORA_POOL_MAX_SIZE", "10")

        mock_backend = MagicMock()
        mock_backend.value = "postgres"

        mock_factory = MagicMock()
        mock_factory.StorageBackend = MagicMock()
        mock_factory.StorageBackend.SQLITE = MagicMock(value="sqlite")
        mock_factory.get_storage_backend = MagicMock(return_value=mock_backend)

        mock_pool = MagicMock()
        mock_pool_manager = MagicMock()
        mock_pool_manager.initialize_shared_pool = AsyncMock(return_value=mock_pool)
        mock_pool_manager.get_pool_info = MagicMock(
            return_value={"pool_size": 10, "is_supabase": False}
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.factory": mock_factory,
                "aragora.storage.pool_manager": mock_pool_manager,
            },
        ):
            from aragora.server.startup.database import init_postgres_pool

            result = await init_postgres_pool()

        assert result["enabled"] is True
        assert result["backend"] == "postgres"
        assert result["pool_size"] == 10
        mock_pool_manager.initialize_shared_pool.assert_awaited_once_with(
            min_size=3,
            max_size=10,
            command_timeout=60.0,
            statement_timeout=60,
        )

    @pytest.mark.asyncio
    async def test_pool_init_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test fallback when pool initialization returns None."""
        monkeypatch.setenv("ARAGORA_USE_SHARED_POOL", "true")

        mock_backend = MagicMock()
        mock_backend.value = "postgres"

        mock_factory = MagicMock()
        mock_factory.StorageBackend = MagicMock()
        mock_factory.StorageBackend.SQLITE = MagicMock(value="sqlite")
        mock_factory.get_storage_backend = MagicMock(return_value=mock_backend)

        mock_pool_manager = MagicMock()
        mock_pool_manager.initialize_shared_pool = AsyncMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.factory": mock_factory,
                "aragora.storage.pool_manager": mock_pool_manager,
            },
        ):
            from aragora.server.startup.database import init_postgres_pool

            result = await init_postgres_pool()

        assert result["enabled"] is False
        assert result["backend"] == "sqlite"
        assert result["reason"] == "pool_init_returned_none"

    @pytest.mark.asyncio
    async def test_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling of connection errors."""
        monkeypatch.setenv("ARAGORA_USE_SHARED_POOL", "true")

        mock_backend = MagicMock()
        mock_backend.value = "postgres"

        mock_factory = MagicMock()
        mock_factory.StorageBackend = MagicMock()
        mock_factory.StorageBackend.SQLITE = MagicMock(value="sqlite")
        mock_factory.get_storage_backend = MagicMock(return_value=mock_backend)

        mock_pool_manager = MagicMock()
        mock_pool_manager.initialize_shared_pool = AsyncMock(
            side_effect=ConnectionError("Database unavailable")
        )

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.factory": mock_factory,
                "aragora.storage.pool_manager": mock_pool_manager,
            },
        ):
            from aragora.server.startup.database import init_postgres_pool

            result = await init_postgres_pool()

        assert result["enabled"] is False
        assert result["backend"] == "sqlite"
        assert "Database unavailable" in result["error"]
        assert result["reason"] == "initialization_failed"

    @pytest.mark.asyncio
    async def test_import_error_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test handling when storage factory is not available."""
        monkeypatch.setenv("ARAGORA_USE_SHARED_POOL", "true")

        with patch.dict("sys.modules", {"aragora.storage.factory": None}):
            import importlib
            import aragora.server.startup.database as db_module

            importlib.reload(db_module)
            result = await db_module.init_postgres_pool()

        assert result["enabled"] is False
        assert result["backend"] == "sqlite"


# =============================================================================
# close_postgres_pool Tests
# =============================================================================


class TestClosePostgresPool:
    """Tests for close_postgres_pool function."""

    @pytest.mark.asyncio
    async def test_successful_close(self) -> None:
        """Test successful pool closure."""
        mock_pool_manager = MagicMock()
        mock_pool_manager.close_shared_pool = AsyncMock()

        with patch.dict("sys.modules", {"aragora.storage.pool_manager": mock_pool_manager}):
            from aragora.server.startup.database import close_postgres_pool

            await close_postgres_pool()

        mock_pool_manager.close_shared_pool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_import_error(self) -> None:
        """Test close handles ImportError gracefully."""
        with patch.dict("sys.modules", {"aragora.storage.pool_manager": None}):
            import importlib
            import aragora.server.startup.database as db_module

            importlib.reload(db_module)
            # Should not raise
            await db_module.close_postgres_pool()

    @pytest.mark.asyncio
    async def test_close_error(self) -> None:
        """Test close handles errors gracefully."""
        mock_pool_manager = MagicMock()
        mock_pool_manager.close_shared_pool = AsyncMock(side_effect=RuntimeError("Close error"))

        with patch.dict("sys.modules", {"aragora.storage.pool_manager": mock_pool_manager}):
            from aragora.server.startup.database import close_postgres_pool

            # Should not raise
            await close_postgres_pool()
