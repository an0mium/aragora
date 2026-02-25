"""Tests for database pool health check functions.

Tests cover:
1. get_database_pool_health() - synchronous pool health
2. check_database_health() - async pool health with SELECT 1 probe
3. Integration with the health endpoint in detailed.py

Verifies:
- Healthy database returns connected: true
- Unreachable database returns connected: false
- High pool utilization triggers warning log
- No pool configured returns not_configured status
- Pool with zero connections returns unhealthy
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.storage.pool_manager import (
    check_database_health,
    get_database_pool_health,
    reset_shared_pool,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_pool_state():
    """Reset pool state before and after each test."""
    reset_shared_pool()
    yield
    reset_shared_pool()


def _make_mock_pool(
    *,
    size: int = 10,
    idle: int = 5,
    max_size: int = 20,
    fetchval_side_effect: object | None = None,
) -> MagicMock:
    """Create a mock asyncpg pool with configurable metrics."""
    pool = MagicMock()
    pool.get_size.return_value = size
    pool.get_idle_size.return_value = idle
    pool.get_max_size.return_value = max_size
    pool.close = AsyncMock()
    pool.terminate = MagicMock()

    mock_conn = MagicMock()
    if fetchval_side_effect is not None:
        mock_conn.fetchval = AsyncMock(side_effect=fetchval_side_effect)
    else:
        mock_conn.fetchval = AsyncMock(return_value=1)

    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_conn
    async_cm.__aexit__.return_value = None
    pool.acquire.return_value = async_cm

    return pool


# =============================================================================
# Test: get_database_pool_health (synchronous)
# =============================================================================


class TestGetDatabasePoolHealth:
    """Tests for the synchronous get_database_pool_health function."""

    def test_not_configured_when_no_pool(self):
        """Returns not_configured when no shared pool exists."""
        result = get_database_pool_health()

        assert result["connected"] is False
        assert result["pool_active"] is None
        assert result["pool_idle"] is None
        assert result["pool_size"] is None
        assert result["pool_utilization_pct"] is None
        assert result["status"] == "not_configured"

    def test_healthy_pool_with_normal_utilization(self):
        """Returns healthy for a pool with < 70% utilization."""
        pool = _make_mock_pool(size=10, idle=5, max_size=20)

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = get_database_pool_health()

        assert result["connected"] is True
        assert result["pool_active"] == 5
        assert result["pool_idle"] == 5
        assert result["pool_size"] == 10
        assert result["pool_utilization_pct"] == 25.0  # 5/20 = 25%
        assert result["status"] == "healthy"

    def test_degraded_when_high_utilization(self, caplog):
        """Returns degraded and logs warning when utilization > 70%."""
        # 18 active out of 20 max = 90% utilization
        pool = _make_mock_pool(size=20, idle=2, max_size=20)

        with (
            patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool),
            caplog.at_level(logging.WARNING, logger="aragora.storage.pool_manager"),
        ):
            result = get_database_pool_health()

        assert result["status"] == "degraded"
        assert result["pool_utilization_pct"] == 90.0  # 18/20
        assert result["pool_active"] == 18
        assert result["connected"] is True

        # Verify warning log was emitted
        assert any("Pool utilization" in record.message for record in caplog.records)

    def test_degraded_at_threshold_boundary(self, caplog):
        """Returns degraded when utilization is exactly at threshold."""
        # 15 active out of 20 max = 75% > 70%
        pool = _make_mock_pool(size=20, idle=5, max_size=20)

        with (
            patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool),
            caplog.at_level(logging.WARNING, logger="aragora.storage.pool_manager"),
        ):
            result = get_database_pool_health()

        assert result["status"] == "degraded"
        assert result["pool_utilization_pct"] == 75.0

    def test_healthy_at_exactly_70_percent(self):
        """Returns healthy when utilization is exactly 70%."""
        # 14 active out of 20 max = 70%
        pool = _make_mock_pool(size=20, idle=6, max_size=20)

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = get_database_pool_health()

        assert result["status"] == "healthy"
        assert result["pool_utilization_pct"] == 70.0

    def test_unhealthy_when_zero_connections(self):
        """Returns unhealthy when pool has zero connections."""
        pool = _make_mock_pool(size=0, idle=0, max_size=20)

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = get_database_pool_health()

        assert result["connected"] is False
        assert result["status"] == "unhealthy"

    def test_pool_without_metric_methods(self):
        """Handles pool objects that lack metric methods gracefully."""
        pool = MagicMock(spec=[])  # No methods at all

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = get_database_pool_health()

        # All metrics should be None since the methods don't exist
        assert result["pool_size"] is None
        assert result["pool_idle"] is None
        assert result["pool_active"] is None
        assert result["pool_utilization_pct"] is None
        # connected is False because pool_size is None (not > 0)
        assert result["connected"] is False

    def test_returns_correct_keys(self):
        """Result dict contains all expected keys."""
        pool = _make_mock_pool()

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = get_database_pool_health()

        expected_keys = {
            "connected",
            "pool_active",
            "pool_idle",
            "pool_size",
            "pool_utilization_pct",
            "status",
        }
        assert set(result.keys()) == expected_keys

    def test_all_connections_idle(self):
        """Healthy with 0% utilization when all connections are idle."""
        pool = _make_mock_pool(size=10, idle=10, max_size=20)

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = get_database_pool_health()

        assert result["pool_active"] == 0
        assert result["pool_utilization_pct"] == 0.0
        assert result["status"] == "healthy"


# =============================================================================
# Test: check_database_health (async)
# =============================================================================


class TestCheckDatabaseHealth:
    """Tests for the async check_database_health function."""

    @pytest.mark.asyncio
    async def test_not_configured_when_no_pool(self):
        """Returns not_configured when no shared pool exists."""
        result = await check_database_health()

        assert result["connected"] is False
        assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    async def test_healthy_database_returns_connected(self):
        """Healthy database returns connected: true."""
        pool = _make_mock_pool(size=10, idle=5, max_size=20)

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = await check_database_health(timeout_seconds=5.0)

        assert result["connected"] is True
        assert result["status"] == "healthy"
        assert result["pool_active"] == 5
        assert result["pool_size"] == 10

    @pytest.mark.asyncio
    async def test_unreachable_database_returns_disconnected(self):
        """Unreachable database returns connected: false."""
        pool = _make_mock_pool(
            size=10,
            idle=5,
            max_size=20,
            fetchval_side_effect=ConnectionError("Connection refused"),
        )

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = await check_database_health(timeout_seconds=5.0)

        assert result["connected"] is False
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_timeout_returns_disconnected(self):
        """Timed-out database probe returns connected: false."""

        async def slow_fetchval(*_args, **_kwargs):
            await asyncio.sleep(10)
            return 1

        pool = _make_mock_pool(size=10, idle=5, max_size=20)

        mock_conn = MagicMock()
        mock_conn.fetchval = slow_fetchval

        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = mock_conn
        async_cm.__aexit__.return_value = None
        pool.acquire.return_value = async_cm

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = await check_database_health(timeout_seconds=0.01)

        assert result["connected"] is False
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_high_utilization_returns_degraded(self, caplog):
        """High pool utilization triggers degraded status and warning log."""
        pool = _make_mock_pool(size=20, idle=2, max_size=20)

        with (
            patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool),
            caplog.at_level(logging.WARNING, logger="aragora.storage.pool_manager"),
        ):
            result = await check_database_health()

        assert result["status"] == "degraded"
        assert result["connected"] is True
        assert result["pool_utilization_pct"] == 90.0

        # Verify warning log was emitted
        assert any("Pool utilization" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_runtime_error_returns_unhealthy(self):
        """RuntimeError from pool returns unhealthy."""
        pool = _make_mock_pool(
            size=10,
            idle=5,
            max_size=20,
            fetchval_side_effect=RuntimeError("Pool is broken"),
        )

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = await check_database_health()

        assert result["connected"] is False
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_os_error_returns_unhealthy(self):
        """OSError from pool returns unhealthy."""
        pool = _make_mock_pool(
            size=10,
            idle=5,
            max_size=20,
            fetchval_side_effect=OSError("Network is unreachable"),
        )

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = await check_database_health()

        assert result["connected"] is False
        assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_returns_pool_metrics_even_on_failure(self):
        """Pool metrics are returned even when the SELECT 1 probe fails."""
        pool = _make_mock_pool(
            size=15,
            idle=10,
            max_size=20,
            fetchval_side_effect=ConnectionError("refused"),
        )

        with patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool):
            result = await check_database_health()

        assert result["connected"] is False
        assert result["pool_size"] == 15
        assert result["pool_idle"] == 10
        assert result["pool_active"] == 5
        assert result["pool_utilization_pct"] == 25.0


# =============================================================================
# Test: Health endpoint integration
# =============================================================================


class TestHealthEndpointIntegration:
    """Tests that database_pool check integrates correctly with health_check."""

    def _make_handler(self, *, tmp_path=None):
        """Create a mock handler for health_check."""
        handler = MagicMock()
        handler.get_storage.return_value = None
        handler.get_elo_system.return_value = None
        handler.get_nomic_dir.return_value = tmp_path
        handler.ctx = {}
        return handler

    def test_health_check_includes_database_pool_healthy(self, tmp_path):
        """Health check includes database_pool when pool is healthy."""
        from aragora.server.handlers.admin.health.detailed import health_check

        pool = _make_mock_pool(size=10, idle=8, max_size=20)
        handler = self._make_handler(tmp_path=tmp_path)

        with (
            patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool),
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "database_pool" in body["checks"]
        db_pool = body["checks"]["database_pool"]
        assert db_pool["connected"] is True
        assert db_pool["healthy"] is True
        assert db_pool["status"] == "healthy"
        assert db_pool["pool_active"] == 2
        assert db_pool["pool_size"] == 10

    def test_health_check_includes_database_pool_not_configured(self, tmp_path):
        """Health check gracefully handles no pool configured."""
        from aragora.server.handlers.admin.health.detailed import health_check

        handler = self._make_handler(tmp_path=tmp_path)

        with (
            patch("aragora.storage.pool_manager.get_shared_pool", return_value=None),
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        assert "database_pool" in body["checks"]
        db_pool = body["checks"]["database_pool"]
        assert db_pool["status"] == "not_configured"
        assert db_pool["healthy"] is True  # not_configured is not a failure

    def test_health_check_pool_degraded_with_warning(self, tmp_path):
        """Health check flags degraded pool with high utilization."""
        from aragora.server.handlers.admin.health.detailed import health_check

        # 18 active out of 20 max = 90%
        pool = _make_mock_pool(size=20, idle=2, max_size=20)
        handler = self._make_handler(tmp_path=tmp_path)

        with (
            patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool),
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        db_pool = body["checks"]["database_pool"]
        assert db_pool["status"] == "degraded"
        assert db_pool["healthy"] is False
        assert db_pool["warning"] == "High pool utilization"

    def test_health_check_pool_unhealthy_with_warning(self, tmp_path):
        """Health check flags unhealthy pool with warning."""
        from aragora.server.handlers.admin.health.detailed import health_check

        pool = _make_mock_pool(size=0, idle=0, max_size=20)
        handler = self._make_handler(tmp_path=tmp_path)

        with (
            patch("aragora.storage.pool_manager.get_shared_pool", return_value=pool),
            patch(
                "aragora.server.handlers.admin.health_utils.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health_utils.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    )
                },
            ),
        ):
            result = health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        db_pool = body["checks"]["database_pool"]
        assert db_pool["status"] == "unhealthy"
        assert db_pool["healthy"] is False
        assert db_pool["warning"] == "Database unreachable"

    def test_health_check_pool_import_error_handled(self, tmp_path):
        """Health check handles ImportError from pool_manager gracefully."""
        from aragora.server.handlers.admin.health.detailed import health_check

        handler = self._make_handler(tmp_path=tmp_path)

        with (
            patch(
                "aragora.server.handlers.admin.health.detailed.check_filesystem_health",
                return_value={"healthy": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_redis_health",
                return_value={"healthy": True, "configured": False},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_ai_providers_health",
                return_value={"healthy": True, "any_available": True},
            ),
            patch(
                "aragora.server.handlers.admin.health.detailed.check_security_services",
                return_value={"healthy": True},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.server.degraded_mode": MagicMock(
                        is_degraded=MagicMock(return_value=False)
                    ),
                    "aragora.storage.pool_manager": None,  # Force ImportError
                },
            ),
        ):
            result = health_check(handler)

        body = json.loads(result.body.decode("utf-8"))
        # Should still return 200, with pool check gracefully degraded
        assert "database_pool" in body["checks"]
        assert body["checks"]["database_pool"]["healthy"] is True
        assert body["checks"]["database_pool"]["status"] == "module_not_available"
