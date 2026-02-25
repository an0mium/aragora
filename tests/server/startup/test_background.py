"""
Tests for aragora.server.startup.background module.

Tests circuit breaker persistence, background tasks, pulse scheduler,
state cleanup, stuck debate watchdog, and Slack token refresh.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# init_circuit_breaker_persistence Tests
# =============================================================================


class TestInitCircuitBreakerPersistence:
    """Tests for init_circuit_breaker_persistence function."""

    def test_successful_initialization(self, tmp_path: Path) -> None:
        """Test successful circuit breaker persistence init."""
        mock_resilience = MagicMock()
        mock_resilience.init_circuit_breaker_persistence = MagicMock()
        mock_resilience.load_circuit_breakers = MagicMock(return_value=5)

        mock_db_config = MagicMock()
        mock_db_config.get_nomic_dir = MagicMock(return_value=tmp_path)

        with patch.dict(
            "sys.modules",
            {
                "aragora.resilience": mock_resilience,
                "aragora.persistence.db_config": mock_db_config,
            },
        ):
            from aragora.server.startup.background import init_circuit_breaker_persistence

            result = init_circuit_breaker_persistence(tmp_path)

        assert result == 5
        mock_resilience.init_circuit_breaker_persistence.assert_called_once()
        mock_resilience.load_circuit_breakers.assert_called_once()

    def test_uses_default_nomic_dir_when_none(self, tmp_path: Path) -> None:
        """Test that default nomic dir is used when None passed."""
        mock_resilience = MagicMock()
        mock_resilience.init_circuit_breaker_persistence = MagicMock()
        mock_resilience.load_circuit_breakers = MagicMock(return_value=0)

        mock_db_config = MagicMock()
        mock_db_config.get_nomic_dir = MagicMock(return_value=tmp_path)

        with patch.dict(
            "sys.modules",
            {
                "aragora.resilience": mock_resilience,
                "aragora.persistence.db_config": mock_db_config,
            },
        ):
            from aragora.server.startup.background import init_circuit_breaker_persistence

            result = init_circuit_breaker_persistence(None)

        assert result == 0
        mock_db_config.get_nomic_dir.assert_called_once()

    def test_import_error_returns_zero(self) -> None:
        """Test ImportError returns 0."""
        with patch.dict("sys.modules", {"aragora.resilience": None}):
            # Force fresh import
            import importlib
            import aragora.server.startup.background as bg_module

            importlib.reload(bg_module)

            # The function should handle ImportError gracefully
            result = bg_module.init_circuit_breaker_persistence(None)
            assert result == 0

    def test_oserror_returns_zero(self, tmp_path: Path) -> None:
        """Test OSError returns 0."""
        mock_resilience = MagicMock()
        mock_resilience.init_circuit_breaker_persistence = MagicMock(
            side_effect=OSError("disk full")
        )
        mock_db_config = MagicMock()
        mock_db_config.get_nomic_dir = MagicMock(return_value=tmp_path)

        with patch.dict(
            "sys.modules",
            {
                "aragora.resilience": mock_resilience,
                "aragora.persistence.db_config": mock_db_config,
            },
        ):
            from aragora.server.startup.background import init_circuit_breaker_persistence

            result = init_circuit_breaker_persistence(tmp_path)

        assert result == 0


# =============================================================================
# init_background_tasks Tests
# =============================================================================


class TestInitBackgroundTasks:
    """Tests for init_background_tasks function."""

    def test_disabled_via_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test background tasks disabled via ARAGORA_DISABLE_BACKGROUND_TASKS."""
        monkeypatch.setenv("ARAGORA_DISABLE_BACKGROUND_TASKS", "1")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        from aragora.server.startup.background import init_background_tasks

        result = init_background_tasks(None)
        assert result is False

    def test_disabled_during_pytest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test background tasks disabled during pytest without test flag."""
        monkeypatch.delenv("ARAGORA_DISABLE_BACKGROUND_TASKS", raising=False)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_something")
        monkeypatch.delenv("ARAGORA_TEST_ENABLE_BACKGROUND_TASKS", raising=False)

        from aragora.server.startup.background import init_background_tasks

        result = init_background_tasks(None)
        assert result is False

    def test_enabled_during_pytest_with_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test background tasks enabled during pytest with test flag."""
        monkeypatch.delenv("ARAGORA_DISABLE_BACKGROUND_TASKS", raising=False)
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_something")
        monkeypatch.setenv("ARAGORA_TEST_ENABLE_BACKGROUND_TASKS", "1")

        mock_background = MagicMock()
        mock_manager = MagicMock()
        mock_background.get_background_manager = MagicMock(return_value=mock_manager)
        mock_background.setup_default_tasks = MagicMock()

        with patch.dict("sys.modules", {"aragora.server.background": mock_background}):
            from aragora.server.startup.background import init_background_tasks

            result = init_background_tasks(tmp_path)

        assert result is True
        mock_manager.start.assert_called_once()

    def test_import_error_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ImportError returns False."""
        monkeypatch.delenv("ARAGORA_DISABLE_BACKGROUND_TASKS", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        with patch.dict("sys.modules", {"aragora.server.background": None}):
            import importlib
            import aragora.server.startup.background as bg_module

            importlib.reload(bg_module)
            result = bg_module.init_background_tasks(None)

        assert result is False


# =============================================================================
# init_pulse_scheduler Tests
# =============================================================================


class TestInitPulseScheduler:
    """Tests for init_pulse_scheduler function."""

    @pytest.mark.asyncio
    async def test_disabled_via_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test pulse scheduler disabled via PULSE_SCHEDULER_AUTOSTART=false."""
        monkeypatch.setenv("PULSE_SCHEDULER_AUTOSTART", "false")

        from aragora.server.startup.background import init_pulse_scheduler

        result = await init_pulse_scheduler()
        assert result is False

    @pytest.mark.asyncio
    async def test_scheduler_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test when scheduler is not available."""
        monkeypatch.setenv("PULSE_SCHEDULER_AUTOSTART", "true")

        mock_pulse = MagicMock()
        mock_pulse.get_pulse_scheduler = MagicMock(return_value=None)

        with patch.dict("sys.modules", {"aragora.server.handlers.pulse": mock_pulse}):
            from aragora.server.startup.background import init_pulse_scheduler

            result = await init_pulse_scheduler()

        assert result is False

    @pytest.mark.asyncio
    async def test_successful_initialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful pulse scheduler initialization."""
        monkeypatch.setenv("PULSE_SCHEDULER_AUTOSTART", "true")
        monkeypatch.setenv("PULSE_SCHEDULER_MAX_PER_HOUR", "10")
        monkeypatch.setenv("PULSE_SCHEDULER_POLL_INTERVAL", "600")

        mock_scheduler = MagicMock()
        mock_scheduler.update_config = MagicMock()
        mock_scheduler.set_debate_creator = MagicMock()
        mock_scheduler.start = AsyncMock()

        mock_pulse = MagicMock()
        mock_pulse.get_pulse_scheduler = MagicMock(return_value=mock_scheduler)

        with patch.dict("sys.modules", {"aragora.server.handlers.pulse": mock_pulse}):
            with patch("asyncio.create_task") as mock_create_task:
                from aragora.server.startup.background import init_pulse_scheduler

                result = await init_pulse_scheduler()

        assert result is True
        mock_scheduler.update_config.assert_called_once_with(
            {
                "poll_interval_seconds": 600,
                "max_debates_per_hour": 10,
            }
        )
        mock_scheduler.set_debate_creator.assert_called_once()
        mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ImportError handling."""
        monkeypatch.setenv("PULSE_SCHEDULER_AUTOSTART", "true")

        with patch.dict("sys.modules", {"aragora.server.handlers.pulse": None}):
            import importlib
            import aragora.server.startup.background as bg_module

            importlib.reload(bg_module)
            result = await bg_module.init_pulse_scheduler()

        assert result is False


# =============================================================================
# init_state_cleanup_task Tests
# =============================================================================


class TestInitStateCleanupTask:
    """Tests for init_state_cleanup_task function."""

    def test_successful_initialization(self) -> None:
        """Test successful state cleanup task initialization."""
        mock_state_manager = MagicMock()
        mock_stream = MagicMock()
        mock_stream.get_stream_state_manager = MagicMock(return_value=mock_state_manager)
        mock_stream.start_cleanup_task = MagicMock()

        with patch.dict("sys.modules", {"aragora.server.stream.state_manager": mock_stream}):
            from aragora.server.startup.background import init_state_cleanup_task

            result = init_state_cleanup_task()

        assert result is True
        mock_stream.start_cleanup_task.assert_called_once_with(
            mock_state_manager, interval_seconds=300
        )

    def test_import_error(self) -> None:
        """Test ImportError returns False."""
        with patch.dict("sys.modules", {"aragora.server.stream.state_manager": None}):
            import importlib
            import aragora.server.startup.background as bg_module

            importlib.reload(bg_module)
            result = bg_module.init_state_cleanup_task()

        assert result is False

    def test_runtime_error(self) -> None:
        """Test RuntimeError returns False."""
        mock_stream = MagicMock()
        mock_stream.get_stream_state_manager = MagicMock(side_effect=RuntimeError("state error"))

        with patch.dict("sys.modules", {"aragora.server.stream.state_manager": mock_stream}):
            from aragora.server.startup.background import init_state_cleanup_task

            result = init_state_cleanup_task()

        assert result is False


# =============================================================================
# init_stuck_debate_watchdog Tests
# =============================================================================


class TestInitStuckDebateWatchdog:
    """Tests for init_stuck_debate_watchdog function."""

    @pytest.mark.asyncio
    async def test_successful_initialization(self) -> None:
        """Test successful watchdog initialization."""
        mock_task = MagicMock(spec=asyncio.Task)
        mock_utils = MagicMock()
        mock_utils.watchdog_stuck_debates = AsyncMock()

        with patch.dict("sys.modules", {"aragora.server.debate_utils": mock_utils}):
            with patch("asyncio.create_task", return_value=mock_task) as mock_create:
                from aragora.server.startup.background import init_stuck_debate_watchdog

                result = await init_stuck_debate_watchdog()

        assert result == mock_task
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_error(self) -> None:
        """Test ImportError returns None."""
        with patch.dict("sys.modules", {"aragora.server.debate_utils": None}):
            import importlib
            import aragora.server.startup.background as bg_module

            importlib.reload(bg_module)
            result = await bg_module.init_stuck_debate_watchdog()

        assert result is None


# =============================================================================
# init_slack_token_refresh_scheduler Tests
# =============================================================================


class TestInitSlackTokenRefreshScheduler:
    """Tests for init_slack_token_refresh_scheduler function."""

    @pytest.mark.asyncio
    async def test_disabled_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test scheduler not started without Slack credentials."""
        monkeypatch.delenv("SLACK_CLIENT_ID", raising=False)
        monkeypatch.delenv("SLACK_CLIENT_SECRET", raising=False)

        from aragora.server.startup.background import init_slack_token_refresh_scheduler

        result = await init_slack_token_refresh_scheduler()
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_with_partial_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test scheduler not started with only client_id."""
        monkeypatch.setenv("SLACK_CLIENT_ID", "test-client-id")
        monkeypatch.delenv("SLACK_CLIENT_SECRET", raising=False)

        from aragora.server.startup.background import init_slack_token_refresh_scheduler

        result = await init_slack_token_refresh_scheduler()
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_initialization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test successful scheduler initialization."""
        monkeypatch.setenv("SLACK_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "test-secret")

        mock_store = MagicMock()
        mock_store.get_expiring_tokens = MagicMock(return_value=[])
        mock_slack = MagicMock()
        mock_slack.get_slack_workspace_store = MagicMock(return_value=mock_store)

        mock_task = MagicMock(spec=asyncio.Task)

        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": mock_slack}):
            with patch("asyncio.create_task", return_value=mock_task) as mock_create:
                from aragora.server.startup.background import (
                    init_slack_token_refresh_scheduler,
                )

                result = await init_slack_token_refresh_scheduler()

        assert result == mock_task
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ImportError returns None."""
        monkeypatch.setenv("SLACK_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "test-secret")

        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            import importlib
            import aragora.server.startup.background as bg_module

            importlib.reload(bg_module)
            result = await bg_module.init_slack_token_refresh_scheduler()

        assert result is None


# =============================================================================
# init_settlement_review_scheduler Tests
# =============================================================================


class TestInitSettlementReviewScheduler:
    """Tests for init_settlement_review_scheduler function."""

    @pytest.mark.asyncio
    async def test_disabled_via_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_SETTLEMENT_REVIEW_ENABLED", "false")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        from aragora.server.startup.background import init_settlement_review_scheduler

        result = await init_settlement_review_scheduler()
        assert result is False

    @pytest.mark.asyncio
    async def test_skips_in_pytest_without_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_settlement")
        monkeypatch.delenv("ARAGORA_TEST_ENABLE_BACKGROUND_TASKS", raising=False)
        from aragora.server.startup.background import init_settlement_review_scheduler

        result = await init_settlement_review_scheduler()
        assert result is False

    @pytest.mark.asyncio
    async def test_starts_scheduler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_SETTLEMENT_REVIEW_ENABLED", "true")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        mock_store_module = MagicMock()
        mock_store = MagicMock()
        mock_store_module.get_receipt_store = MagicMock(return_value=mock_store)

        mock_scheduler_module = MagicMock()
        mock_scheduler = MagicMock()
        mock_scheduler.is_running = False
        mock_scheduler.start = AsyncMock()
        mock_scheduler_module.get_settlement_review_scheduler = MagicMock(return_value=mock_scheduler)

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.receipt_store": mock_store_module,
                "aragora.scheduler.settlement_review": mock_scheduler_module,
            },
        ):
            from aragora.server.startup.background import init_settlement_review_scheduler

            result = await init_settlement_review_scheduler()

        assert result is True
        mock_scheduler.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_import_error_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ARAGORA_SETTLEMENT_REVIEW_ENABLED", "true")
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

        with patch.dict(
            "sys.modules",
            {
                "aragora.storage.receipt_store": None,
                "aragora.scheduler.settlement_review": None,
            },
        ):
            import importlib
            import aragora.server.startup.background as bg_module

            importlib.reload(bg_module)
            result = await bg_module.init_settlement_review_scheduler()

        assert result is False
