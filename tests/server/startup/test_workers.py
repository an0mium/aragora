"""Tests for server startup worker initialization.

Tests the SLO webhooks, webhook dispatcher, gauntlet recovery,
durable job queue, workers, and backup scheduler initialization.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.startup.workers import (
    get_gauntlet_worker,
    get_testfixer_task_worker,
    get_testfixer_worker,
    init_backup_scheduler,
    init_durable_job_queue_recovery,
    init_gauntlet_run_recovery,
    init_gauntlet_worker,
    init_notification_worker,
    init_slo_webhooks,
    init_testfixer_task_worker,
    init_testfixer_worker,
    init_webhook_dispatcher,
    init_workflow_checkpoint_persistence,
)


# =============================================================================
# Test: Worker getters
# =============================================================================


class TestWorkerGetters:
    """Tests for worker getter functions."""

    def test_get_gauntlet_worker_returns_none_initially(self):
        """Test that gauntlet worker is None before initialization."""
        # Reset module state
        import aragora.server.startup.workers as workers_module

        workers_module._gauntlet_worker = None
        result = get_gauntlet_worker()
        assert result is None

    def test_get_testfixer_worker_returns_none_initially(self):
        """Test that testfixer worker is None before initialization."""
        import aragora.server.startup.workers as workers_module

        workers_module._testfixer_worker = None
        result = get_testfixer_worker()
        assert result is None

    def test_get_testfixer_task_worker_returns_none_initially(self):
        """Test that testfixer task worker is None before initialization."""
        import aragora.server.startup.workers as workers_module

        workers_module._testfixer_task_worker = None
        result = get_testfixer_task_worker()
        assert result is None


# =============================================================================
# Test: init_slo_webhooks
# =============================================================================


class TestInitSloWebhooks:
    """Tests for SLO webhook initialization."""

    def test_returns_true_on_success(self):
        """Test that function returns True on successful initialization."""
        mock_module = MagicMock()
        mock_module.init_slo_webhooks = MagicMock(return_value=True)

        with patch.dict(
            "sys.modules",
            {"aragora.observability.metrics.slo": mock_module},
        ):
            result = init_slo_webhooks()

        assert result is True

    def test_returns_false_when_dispatcher_unavailable(self):
        """Test that function returns False when dispatcher is unavailable."""
        mock_module = MagicMock()
        mock_module.init_slo_webhooks = MagicMock(return_value=False)

        with patch.dict(
            "sys.modules",
            {"aragora.observability.metrics.slo": mock_module},
        ):
            result = init_slo_webhooks()

        assert result is False

    def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            "sys.modules",
            {"aragora.observability.metrics.slo": None},
        ):
            result = init_slo_webhooks()

        assert result is False

    def test_runtime_error_returns_false(self):
        """Test that RuntimeError returns False."""
        mock_module = MagicMock()
        mock_module.init_slo_webhooks = MagicMock(side_effect=RuntimeError("Init failed"))

        with patch.dict(
            "sys.modules",
            {"aragora.observability.metrics.slo": mock_module},
        ):
            result = init_slo_webhooks()

        assert result is False


# =============================================================================
# Test: init_webhook_dispatcher
# =============================================================================


class TestInitWebhookDispatcher:
    """Tests for webhook dispatcher initialization."""

    def test_returns_true_with_configs(self):
        """Test that function returns True when configs are found."""
        mock_dispatcher = MagicMock()
        mock_dispatcher.configs = [MagicMock(), MagicMock()]

        mock_module = MagicMock()
        mock_module.init_dispatcher = MagicMock(return_value=mock_dispatcher)

        with patch.dict(
            "sys.modules",
            {"aragora.integrations.webhooks": mock_module},
        ):
            result = init_webhook_dispatcher()

        assert result is True

    def test_returns_false_without_configs(self):
        """Test that function returns False when no configs found."""
        mock_module = MagicMock()
        mock_module.init_dispatcher = MagicMock(return_value=None)

        with patch.dict(
            "sys.modules",
            {"aragora.integrations.webhooks": mock_module},
        ):
            result = init_webhook_dispatcher()

        assert result is False

    def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            "sys.modules",
            {"aragora.integrations.webhooks": None},
        ):
            result = init_webhook_dispatcher()

        assert result is False


# =============================================================================
# Test: init_gauntlet_run_recovery
# =============================================================================


class TestInitGauntletRunRecovery:
    """Tests for gauntlet run recovery initialization."""

    def test_returns_count_on_success(self):
        """Test that function returns recovery count."""
        mock_module = MagicMock()
        mock_module.recover_stale_gauntlet_runs = MagicMock(return_value=5)

        with patch.dict(
            os.environ,
            {"ARAGORA_DISABLE_GAUNTLET_RECOVERY": ""},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.server.handlers.gauntlet": mock_module},
            ):
                result = init_gauntlet_run_recovery()

        assert result == 5

    def test_disabled_by_env_returns_zero(self):
        """Test that function returns 0 when disabled."""
        with patch.dict(
            os.environ,
            {"ARAGORA_DISABLE_GAUNTLET_RECOVERY": "true"},
            clear=False,
        ):
            result = init_gauntlet_run_recovery()

        assert result == 0

    def test_import_error_returns_zero(self):
        """Test that ImportError returns 0."""
        with patch.dict(
            os.environ,
            {"ARAGORA_DISABLE_GAUNTLET_RECOVERY": ""},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.server.handlers.gauntlet": None},
            ):
                result = init_gauntlet_run_recovery()

        assert result == 0


# =============================================================================
# Test: init_durable_job_queue_recovery
# =============================================================================


class TestInitDurableJobQueueRecovery:
    """Tests for durable job queue recovery."""

    @pytest.mark.asyncio
    async def test_returns_count_on_success(self):
        """Test that function returns recovery count."""
        mock_module = MagicMock()
        mock_module.recover_interrupted_gauntlets = AsyncMock(return_value=3)

        with patch.dict(
            os.environ,
            {"ARAGORA_DURABLE_GAUNTLET": "1"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.queue.workers.gauntlet_worker": mock_module},
            ):
                result = await init_durable_job_queue_recovery()

        assert result == 3

    @pytest.mark.asyncio
    async def test_disabled_by_env_returns_zero(self):
        """Test that function returns 0 when disabled."""
        with patch.dict(
            os.environ,
            {"ARAGORA_DURABLE_GAUNTLET": "0"},
            clear=False,
        ):
            result = await init_durable_job_queue_recovery()

        assert result == 0

    @pytest.mark.asyncio
    async def test_import_error_returns_zero(self):
        """Test that ImportError returns 0."""
        with patch.dict(
            os.environ,
            {"ARAGORA_DURABLE_GAUNTLET": "1"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.queue.workers.gauntlet_worker": None},
            ):
                result = await init_durable_job_queue_recovery()

        assert result == 0


# =============================================================================
# Test: init_gauntlet_worker
# =============================================================================


class TestInitGauntletWorker:
    """Tests for gauntlet worker initialization."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self):
        """Test that function returns True on success."""
        mock_worker = MagicMock()
        mock_worker.start = AsyncMock()

        mock_module = MagicMock()
        mock_module.GauntletWorker = MagicMock(return_value=mock_worker)

        with patch.dict(
            os.environ,
            {"ARAGORA_DURABLE_GAUNTLET": "1", "ARAGORA_GAUNTLET_WORKERS": "3"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.queue.workers.gauntlet_worker": mock_module},
            ):
                result = await init_gauntlet_worker()

        assert result is True

    @pytest.mark.asyncio
    async def test_disabled_by_env_returns_false(self):
        """Test that function returns False when disabled."""
        with patch.dict(
            os.environ,
            {"ARAGORA_DURABLE_GAUNTLET": "0"},
            clear=False,
        ):
            result = await init_gauntlet_worker()

        assert result is False

    @pytest.mark.asyncio
    async def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            os.environ,
            {"ARAGORA_DURABLE_GAUNTLET": "1"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.queue.workers.gauntlet_worker": None},
            ):
                result = await init_gauntlet_worker()

        assert result is False


# =============================================================================
# Test: init_notification_worker
# =============================================================================


class TestInitNotificationWorker:
    """Tests for notification worker initialization."""

    @pytest.mark.asyncio
    async def test_disabled_by_env_returns_false(self):
        """Test that function returns False when disabled."""
        with patch.dict(
            os.environ,
            {"ARAGORA_NOTIFICATION_WORKER": "0"},
            clear=False,
        ):
            result = await init_notification_worker()

        assert result is False

    @pytest.mark.asyncio
    async def test_no_redis_url_returns_false(self):
        """Test that function returns False without Redis URL."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_NOTIFICATION_WORKER": "1",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
            },
            clear=False,
        ):
            result = await init_notification_worker()

        assert result is False

    @pytest.mark.asyncio
    async def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_NOTIFICATION_WORKER": "1",
                "REDIS_URL": "redis://localhost:6379",
            },
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"redis.asyncio": None},
            ):
                result = await init_notification_worker()

        assert result is False


# =============================================================================
# Test: init_testfixer_worker
# =============================================================================


class TestInitTestfixerWorker:
    """Tests for testfixer worker initialization."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self):
        """Test that function returns True on success."""
        mock_worker = MagicMock()
        mock_worker.start = AsyncMock()

        mock_module = MagicMock()
        mock_module.TestFixerWorker = MagicMock(return_value=mock_worker)

        with patch.dict(
            os.environ,
            {"ARAGORA_TESTFIXER_WORKER": "1"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.queue.workers.testfixer_worker": mock_module},
            ):
                result = await init_testfixer_worker()

        assert result is True

    @pytest.mark.asyncio
    async def test_disabled_by_env_returns_false(self):
        """Test that function returns False when disabled."""
        with patch.dict(
            os.environ,
            {"ARAGORA_TESTFIXER_WORKER": "0"},
            clear=False,
        ):
            result = await init_testfixer_worker()

        assert result is False

    @pytest.mark.asyncio
    async def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            os.environ,
            {"ARAGORA_TESTFIXER_WORKER": "1"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.queue.workers.testfixer_worker": None},
            ):
                result = await init_testfixer_worker()

        assert result is False


# =============================================================================
# Test: init_testfixer_task_worker
# =============================================================================


class TestInitTestfixerTaskWorker:
    """Tests for testfixer task worker initialization."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """Test that function is disabled by default."""
        with patch.dict(
            os.environ,
            {"ARAGORA_TESTFIXER_TASK_WORKER": "0"},
            clear=False,
        ):
            result = await init_testfixer_task_worker()

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_enabled(self):
        """Test that function returns True when enabled."""
        mock_worker = MagicMock()
        mock_module = MagicMock()
        mock_module.start_testfixer_worker = AsyncMock(return_value=mock_worker)

        with patch.dict(
            os.environ,
            {"ARAGORA_TESTFIXER_TASK_WORKER": "1"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.nomic.testfixer.worker_loop": mock_module},
            ):
                result = await init_testfixer_task_worker()

        assert result is True

    @pytest.mark.asyncio
    async def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            os.environ,
            {"ARAGORA_TESTFIXER_TASK_WORKER": "1"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.nomic.testfixer.worker_loop": None},
            ):
                result = await init_testfixer_task_worker()

        assert result is False


# =============================================================================
# Test: init_workflow_checkpoint_persistence
# =============================================================================


class TestInitWorkflowCheckpointPersistence:
    """Tests for workflow checkpoint persistence initialization."""

    def test_returns_true_on_success(self):
        """Test that function returns True on success."""
        mock_mound = MagicMock()

        mock_km_module = MagicMock()
        mock_km_module.get_knowledge_mound = MagicMock(return_value=mock_mound)

        mock_checkpoint_module = MagicMock()
        mock_checkpoint_module.set_default_knowledge_mound = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "aragora.knowledge.mound": mock_km_module,
                "aragora.workflow.checkpoint_store": mock_checkpoint_module,
            },
        ):
            result = init_workflow_checkpoint_persistence()

        assert result is True
        mock_checkpoint_module.set_default_knowledge_mound.assert_called_once_with(mock_mound)

    def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound": None},
        ):
            result = init_workflow_checkpoint_persistence()

        assert result is False

    def test_runtime_error_returns_false(self):
        """Test that RuntimeError returns False."""
        mock_km_module = MagicMock()
        mock_km_module.get_knowledge_mound = MagicMock(
            side_effect=RuntimeError("Mound init failed")
        )

        with patch.dict(
            "sys.modules",
            {"aragora.knowledge.mound": mock_km_module},
        ):
            result = init_workflow_checkpoint_persistence()

        assert result is False


# =============================================================================
# Test: init_backup_scheduler
# =============================================================================


class TestInitBackupScheduler:
    """Tests for backup scheduler initialization."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """Test that scheduler is disabled by default."""
        with patch.dict(
            os.environ,
            {"BACKUP_ENABLED": "false"},
            clear=False,
        ):
            result = await init_backup_scheduler()

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_enabled(self):
        """Test that function returns True when enabled."""
        mock_manager = MagicMock()

        mock_backup_module = MagicMock()
        mock_backup_module.get_backup_manager = MagicMock(return_value=mock_manager)

        mock_scheduler_module = MagicMock()
        mock_scheduler_module.BackupSchedule = MagicMock()
        mock_scheduler_module.start_backup_scheduler = AsyncMock()

        with patch.dict(
            os.environ,
            {
                "BACKUP_ENABLED": "true",
                "BACKUP_DAILY_TIME": "02:00",
                "BACKUP_DR_DRILL_ENABLED": "true",
                "BACKUP_DR_DRILL_INTERVAL_DAYS": "30",
            },
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.backup.manager": mock_backup_module,
                    "aragora.backup.scheduler": mock_scheduler_module,
                },
            ):
                result = await init_backup_scheduler()

        assert result is True
        mock_scheduler_module.start_backup_scheduler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_time_uses_default(self):
        """Test that invalid time format uses default."""
        mock_manager = MagicMock()

        mock_backup_module = MagicMock()
        mock_backup_module.get_backup_manager = MagicMock(return_value=mock_manager)

        mock_scheduler_module = MagicMock()
        mock_scheduler_module.BackupSchedule = MagicMock()
        mock_scheduler_module.start_backup_scheduler = AsyncMock()

        with patch.dict(
            os.environ,
            {
                "BACKUP_ENABLED": "true",
                "BACKUP_DAILY_TIME": "invalid",
            },
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {
                    "aragora.backup.manager": mock_backup_module,
                    "aragora.backup.scheduler": mock_scheduler_module,
                },
            ):
                result = await init_backup_scheduler()

        assert result is True

    @pytest.mark.asyncio
    async def test_import_error_returns_false(self):
        """Test that ImportError returns False."""
        with patch.dict(
            os.environ,
            {"BACKUP_ENABLED": "true"},
            clear=False,
        ):
            with patch.dict(
                "sys.modules",
                {"aragora.backup.manager": None},
            ):
                result = await init_backup_scheduler()

        assert result is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestWorkersStartupIntegration:
    """Integration tests for workers startup sequence."""

    def test_environment_variable_parsing(self):
        """Test that environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(
                os.environ,
                {"BACKUP_ENABLED": env_value},
                clear=False,
            ):
                enabled = os.environ.get("BACKUP_ENABLED", "false").lower() in (
                    "true",
                    "1",
                    "yes",
                )
                assert enabled == expected, f"Failed for value: {env_value}"

    def test_worker_getters_return_set_values(self):
        """Test that worker getters return set values."""
        import aragora.server.startup.workers as workers_module

        # Test gauntlet worker
        mock_gauntlet = MagicMock()
        workers_module._gauntlet_worker = mock_gauntlet
        assert get_gauntlet_worker() is mock_gauntlet

        # Test testfixer worker
        mock_testfixer = MagicMock()
        workers_module._testfixer_worker = mock_testfixer
        assert get_testfixer_worker() is mock_testfixer

        # Test testfixer task worker
        mock_task = MagicMock()
        workers_module._testfixer_task_worker = mock_task
        assert get_testfixer_task_worker() is mock_task

        # Clean up
        workers_module._gauntlet_worker = None
        workers_module._testfixer_worker = None
        workers_module._testfixer_task_worker = None
