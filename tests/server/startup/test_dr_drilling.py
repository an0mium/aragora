"""
Tests for aragora.server.startup.dr_drilling module.

Tests DR drill scheduler startup and shutdown.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# get_dr_drill_scheduler Tests
# =============================================================================


class TestGetDRDrillScheduler:
    """Tests for get_dr_drill_scheduler function."""

    def test_initially_none(self) -> None:
        """Test get_dr_drill_scheduler returns None initially."""
        import aragora.server.startup.dr_drilling as dr_module

        # Reset global state
        dr_module._dr_drill_scheduler = None

        result = dr_module.get_dr_drill_scheduler()
        assert result is None

    def test_returns_scheduler_when_set(self) -> None:
        """Test get_dr_drill_scheduler returns scheduler when set."""
        import aragora.server.startup.dr_drilling as dr_module

        mock_scheduler = MagicMock()
        dr_module._dr_drill_scheduler = mock_scheduler

        result = dr_module.get_dr_drill_scheduler()
        assert result == mock_scheduler

        # Cleanup
        dr_module._dr_drill_scheduler = None


# =============================================================================
# start_dr_drilling Tests
# =============================================================================


class TestStartDRDrilling:
    """Tests for start_dr_drilling function."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test DR drilling disabled by default."""
        monkeypatch.delenv("ARAGORA_DR_DRILL_ENABLED", raising=False)

        from aragora.server.startup.dr_drilling import start_dr_drilling

        result = await start_dr_drilling()
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_with_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test DR drilling disabled with ARAGORA_DR_DRILL_ENABLED=false."""
        monkeypatch.setenv("ARAGORA_DR_DRILL_ENABLED", "false")

        from aragora.server.startup.dr_drilling import start_dr_drilling

        result = await start_dr_drilling()
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful DR drill scheduler start."""
        monkeypatch.setenv("ARAGORA_DR_DRILL_ENABLED", "true")
        monkeypatch.setenv("ARAGORA_DR_DRILL_MONTHLY_DAY", "20")
        monkeypatch.setenv("ARAGORA_DR_DRILL_TARGET_RTO_SECONDS", "1800")
        monkeypatch.setenv("ARAGORA_DR_DRILL_TARGET_RPO_SECONDS", "600")
        monkeypatch.setenv("ARAGORA_DR_DRILL_DRY_RUN", "true")

        mock_scheduler = MagicMock()
        mock_scheduler.start = AsyncMock()

        mock_config = MagicMock()

        mock_dr_module = MagicMock()
        mock_dr_module.DRDrillConfig = MagicMock(return_value=mock_config)
        mock_dr_module.DRDrillScheduler = MagicMock(return_value=mock_scheduler)

        with patch.dict("sys.modules", {"aragora.scheduler.dr_drill_scheduler": mock_dr_module}):
            import aragora.server.startup.dr_drilling as dr_startup

            # Reset global state
            dr_startup._dr_drill_scheduler = None

            result = await dr_startup.start_dr_drilling()

        assert result == mock_scheduler
        mock_scheduler.start.assert_awaited_once()
        mock_dr_module.DRDrillConfig.assert_called_once_with(
            monthly_drill_day=20,
            quarterly_drill_months=[3, 6, 9, 12],
            annual_drill_month=1,
            target_rto_seconds=1800.0,
            target_rpo_seconds=600.0,
            storage_path=None,
            dry_run=True,
        )

    @pytest.mark.asyncio
    async def test_start_with_storage_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test DR drill scheduler start with storage path."""
        monkeypatch.setenv("ARAGORA_DR_DRILL_ENABLED", "true")
        monkeypatch.setenv("ARAGORA_DR_DRILL_STORAGE_PATH", "/tmp/dr_drills")

        mock_scheduler = MagicMock()
        mock_scheduler.start = AsyncMock()

        mock_dr_module = MagicMock()
        mock_dr_module.DRDrillConfig = MagicMock()
        mock_dr_module.DRDrillScheduler = MagicMock(return_value=mock_scheduler)

        with patch.dict("sys.modules", {"aragora.scheduler.dr_drill_scheduler": mock_dr_module}):
            import aragora.server.startup.dr_drilling as dr_startup

            dr_startup._dr_drill_scheduler = None
            result = await dr_startup.start_dr_drilling()

        assert result == mock_scheduler
        call_kwargs = mock_dr_module.DRDrillConfig.call_args[1]
        assert call_kwargs["storage_path"] == "/tmp/dr_drills"

    @pytest.mark.asyncio
    async def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ImportError handling."""
        monkeypatch.setenv("ARAGORA_DR_DRILL_ENABLED", "true")

        with patch.dict("sys.modules", {"aragora.scheduler.dr_drill_scheduler": None}):
            import importlib
            import aragora.server.startup.dr_drilling as dr_module

            importlib.reload(dr_module)
            result = await dr_module.start_dr_drilling()

        assert result is None

    @pytest.mark.asyncio
    async def test_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test RuntimeError handling."""
        monkeypatch.setenv("ARAGORA_DR_DRILL_ENABLED", "true")

        mock_dr_module = MagicMock()
        mock_dr_module.DRDrillConfig = MagicMock(side_effect=RuntimeError("Config error"))

        with patch.dict("sys.modules", {"aragora.scheduler.dr_drill_scheduler": mock_dr_module}):
            from aragora.server.startup.dr_drilling import start_dr_drilling

            result = await start_dr_drilling()

        assert result is None


# =============================================================================
# stop_dr_drilling Tests
# =============================================================================


class TestStopDRDrilling:
    """Tests for stop_dr_drilling function."""

    @pytest.mark.asyncio
    async def test_stop_when_none(self) -> None:
        """Test stop does nothing when scheduler is None."""
        import aragora.server.startup.dr_drilling as dr_module

        dr_module._dr_drill_scheduler = None

        # Should not raise
        await dr_module.stop_dr_drilling()

        assert dr_module._dr_drill_scheduler is None

    @pytest.mark.asyncio
    async def test_successful_stop(self) -> None:
        """Test successful scheduler stop."""
        import aragora.server.startup.dr_drilling as dr_module

        mock_scheduler = MagicMock()
        mock_scheduler.stop = AsyncMock()
        dr_module._dr_drill_scheduler = mock_scheduler

        await dr_module.stop_dr_drilling()

        mock_scheduler.stop.assert_awaited_once()
        assert dr_module._dr_drill_scheduler is None

    @pytest.mark.asyncio
    async def test_stop_with_error(self) -> None:
        """Test stop handles errors gracefully."""
        import aragora.server.startup.dr_drilling as dr_module

        mock_scheduler = MagicMock()
        mock_scheduler.stop = AsyncMock(side_effect=RuntimeError("Stop error"))
        dr_module._dr_drill_scheduler = mock_scheduler

        # Should not raise
        await dr_module.stop_dr_drilling()

        mock_scheduler.stop.assert_awaited_once()
        assert dr_module._dr_drill_scheduler is None
