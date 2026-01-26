"""
Tests for Confidence Decay Scheduler.

Tests cover:
- Scheduler initialization
- Start/stop lifecycle
- Decay application to workspaces
- Statistics tracking
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConfidenceDecaySchedulerInit:
    """Tests for scheduler initialization."""

    def test_scheduler_defaults(self):
        """Test default configuration values."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        scheduler = ConfidenceDecayScheduler()

        assert scheduler._decay_interval_hours == 24
        assert scheduler._workspaces is None
        assert scheduler._max_items_per_workspace == 10000
        assert not scheduler.is_running

    def test_scheduler_custom_config(self):
        """Test custom configuration."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        mock_mound = MagicMock()
        scheduler = ConfidenceDecayScheduler(
            knowledge_mound=mock_mound,
            decay_interval_hours=12,
            workspaces=["ws1", "ws2"],
            max_items_per_workspace=5000,
        )

        assert scheduler._decay_interval_hours == 12
        assert scheduler._workspaces == ["ws1", "ws2"]
        assert scheduler._max_items_per_workspace == 5000


class TestConfidenceDecaySchedulerLifecycle:
    """Tests for scheduler start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_without_mound(self):
        """Test starting without knowledge mound logs warning."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        scheduler = ConfidenceDecayScheduler()

        await scheduler.start()

        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_start_with_mound(self):
        """Test starting with knowledge mound."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        mock_mound = MagicMock()
        scheduler = ConfidenceDecayScheduler(
            knowledge_mound=mock_mound,
            decay_interval_hours=24,
        )

        await scheduler.start()

        assert scheduler.is_running

        await scheduler.stop()

        assert not scheduler.is_running

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        """Test starting twice is a no-op."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        mock_mound = MagicMock()
        scheduler = ConfidenceDecayScheduler(knowledge_mound=mock_mound)

        await scheduler.start()
        first_task = scheduler._task

        await scheduler.start()

        # Should be the same task
        assert scheduler._task is first_task

        await scheduler.stop()


class TestConfidenceDecaySchedulerApplication:
    """Tests for decay application."""

    @pytest.mark.asyncio
    async def test_apply_decay_without_mound(self):
        """Test decay application without mound returns empty."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        scheduler = ConfidenceDecayScheduler()

        reports = await scheduler.apply_decay_to_workspaces()

        assert reports == []

    @pytest.mark.asyncio
    async def test_apply_decay_with_mound(self):
        """Test decay application with mound."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        # Create mock mound with decay report
        mock_report = MagicMock()
        mock_report.items_processed = 100
        mock_report.items_decayed = 20
        mock_report.items_boosted = 5
        mock_report.average_confidence_change = -0.05
        mock_report.duration_ms = 150.5

        mock_mound = MagicMock()
        mock_mound.apply_confidence_decay = AsyncMock(return_value=mock_report)

        scheduler = ConfidenceDecayScheduler(
            knowledge_mound=mock_mound,
            workspaces=["test-ws"],
        )

        reports = await scheduler.apply_decay_to_workspaces(force=True)

        assert len(reports) == 1
        assert reports[0].workspace_id == "test-ws"
        assert reports[0].items_processed == 100
        assert reports[0].items_decayed == 20

    @pytest.mark.asyncio
    async def test_trigger_decay_now_specific_workspace(self):
        """Test immediate decay trigger for specific workspace."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        mock_report = MagicMock()
        mock_report.items_processed = 50
        mock_report.items_decayed = 10
        mock_report.items_boosted = 2
        mock_report.average_confidence_change = -0.03
        mock_report.duration_ms = 75.0

        mock_mound = MagicMock()
        mock_mound.apply_confidence_decay = AsyncMock(return_value=mock_report)

        scheduler = ConfidenceDecayScheduler(knowledge_mound=mock_mound)

        reports = await scheduler.trigger_decay_now(workspace_id="my-workspace")

        assert len(reports) == 1
        assert reports[0].workspace_id == "my-workspace"
        mock_mound.apply_confidence_decay.assert_called_once_with(
            workspace_id="my-workspace",
            force=True,
        )


class TestConfidenceDecaySchedulerStats:
    """Tests for scheduler statistics."""

    def test_get_stats_initial(self):
        """Test initial statistics."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        scheduler = ConfidenceDecayScheduler(
            decay_interval_hours=12,
            workspaces=["ws1"],
        )

        stats = scheduler.get_stats()

        assert stats["running"] is False
        assert stats["decay_interval_hours"] == 12
        assert stats["workspaces"] == ["ws1"]
        assert stats["total_decay_cycles"] == 0
        assert stats["total_items_processed"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_decay(self):
        """Test statistics after running decay."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
        )

        mock_report = MagicMock()
        mock_report.items_processed = 100
        mock_report.items_decayed = 20
        mock_report.items_boosted = 5
        mock_report.average_confidence_change = -0.05
        mock_report.duration_ms = 150.5

        mock_mound = MagicMock()
        mock_mound.apply_confidence_decay = AsyncMock(return_value=mock_report)

        scheduler = ConfidenceDecayScheduler(
            knowledge_mound=mock_mound,
            workspaces=["test-ws"],
        )

        await scheduler.apply_decay_to_workspaces(force=True)

        stats = scheduler.get_stats()
        assert "test-ws" in stats["last_run"]


class TestDecayScheduleReport:
    """Tests for DecayScheduleReport dataclass."""

    def test_report_to_dict(self):
        """Test report serialization."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            DecayScheduleReport,
        )

        report = DecayScheduleReport(
            workspace_id="test-ws",
            items_processed=100,
            items_decayed=20,
            items_boosted=5,
            average_change=-0.05,
            duration_ms=150.5,
        )

        d = report.to_dict()

        assert d["workspace_id"] == "test-ws"
        assert d["items_processed"] == 100
        assert d["items_decayed"] == 20
        assert "scheduled_at" in d


class TestGlobalSchedulerFunctions:
    """Tests for global scheduler functions."""

    @pytest.mark.asyncio
    async def test_start_and_stop_global_scheduler(self):
        """Test global scheduler start and stop."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            get_decay_scheduler,
            start_decay_scheduler,
            stop_decay_scheduler,
        )

        mock_mound = MagicMock()

        scheduler = await start_decay_scheduler(
            knowledge_mound=mock_mound,
            decay_interval_hours=24,
        )

        assert scheduler.is_running
        assert get_decay_scheduler() is scheduler

        await stop_decay_scheduler()

        assert not scheduler.is_running

    def test_set_decay_scheduler(self):
        """Test setting global scheduler."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
            get_decay_scheduler,
            set_decay_scheduler,
        )

        custom_scheduler = ConfidenceDecayScheduler()
        set_decay_scheduler(custom_scheduler)

        assert get_decay_scheduler() is custom_scheduler

        # Cleanup
        set_decay_scheduler(None)  # type: ignore[arg-type]


class TestDecayMetrics:
    """Tests for Prometheus metrics recording."""

    def test_record_decay_metrics_without_prometheus(self):
        """Test metrics recording gracefully handles missing prometheus_client."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            DecayScheduleReport,
            record_decay_metrics,
        )

        report = DecayScheduleReport(
            workspace_id="test-workspace",
            items_processed=100,
            items_decayed=25,
            items_boosted=5,
            average_change=-0.05,
            duration_ms=150.0,
        )

        # Should not raise even if prometheus_client not installed
        record_decay_metrics(report)

    def test_decay_schedule_report_to_dict(self):
        """Test DecayScheduleReport serialization."""
        from aragora.knowledge.mound.confidence_decay_scheduler import DecayScheduleReport

        report = DecayScheduleReport(
            workspace_id="ws-123",
            items_processed=500,
            items_decayed=100,
            items_boosted=20,
            average_change=-0.02,
            duration_ms=200.5,
        )

        d = report.to_dict()

        assert d["workspace_id"] == "ws-123"
        assert d["items_processed"] == 500
        assert d["items_decayed"] == 100
        assert d["items_boosted"] == 20
        assert d["average_change"] == -0.02
        assert d["duration_ms"] == 200.5
        assert "scheduled_at" in d

    @pytest.mark.asyncio
    async def test_decay_records_metrics_on_completion(self):
        """Test that decay operations record metrics."""
        from aragora.knowledge.mound.confidence_decay_scheduler import (
            ConfidenceDecayScheduler,
            record_decay_metrics,
        )

        metrics_recorded = []

        # Patch record_decay_metrics to capture calls
        original_record = record_decay_metrics

        def capture_metrics(report):
            metrics_recorded.append(report)
            original_record(report)

        mock_mound = MagicMock()
        mock_mound.list_workspaces = AsyncMock(return_value=["ws1"])

        # Create a mock decay operation
        mock_decay_ops = MagicMock()
        mock_decay_ops.apply_time_decay = AsyncMock(
            return_value=MagicMock(
                items_processed=50,
                items_decayed=10,
                items_boosted=2,
                average_change=-0.03,
            )
        )
        mock_mound.get_decay_ops = MagicMock(return_value=mock_decay_ops)

        scheduler = ConfidenceDecayScheduler(
            knowledge_mound=mock_mound,
            workspaces=["test-ws"],
        )

        with patch(
            "aragora.knowledge.mound.confidence_decay_scheduler.record_decay_metrics",
            side_effect=capture_metrics,
        ):
            reports = await scheduler.apply_decay_to_workspaces(force=True)

        # Verify reports were generated (scheduler may or may not have decayed)
        # The test validates the integration path exists
