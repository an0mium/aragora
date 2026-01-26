"""
Tests for StartupTransaction with rollback support.

Tests cover:
- Basic transaction lifecycle (enter/exit)
- Cleanup registration and rollback on failure
- SLO tracking and warning
- Checkpoint creation
- Report generation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestStartupTransactionBasics:
    """Basic tests for StartupTransaction."""

    @pytest.mark.asyncio
    async def test_successful_startup(self):
        """Test successful startup transaction."""
        from aragora.server.startup_transaction import StartupTransaction

        async with StartupTransaction(slo_seconds=10) as txn:
            txn.mark_initialized("component_a")
            txn.mark_initialized("component_b")

        report = txn.get_report()
        assert report.success is True
        assert report.components_initialized == 2
        assert len(report.components_failed) == 0

    @pytest.mark.asyncio
    async def test_failed_startup_runs_cleanups(self):
        """Test that cleanups run on failure."""
        from aragora.server.startup_transaction import StartupTransaction

        cleanup_calls = []

        def cleanup_a():
            cleanup_calls.append("a")

        async def cleanup_b():
            cleanup_calls.append("b")

        try:
            async with StartupTransaction() as txn:
                txn.register_cleanup("a", cleanup_a)
                txn.mark_initialized("a")

                txn.register_cleanup("b", cleanup_b)
                txn.mark_initialized("b")

                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        # Cleanups should run in reverse order
        assert cleanup_calls == ["b", "a"]

    @pytest.mark.asyncio
    async def test_cleanups_discarded_on_success(self):
        """Test that cleanups are discarded on successful completion."""
        from aragora.server.startup_transaction import StartupTransaction

        cleanup_calls = []

        def cleanup_a():
            cleanup_calls.append("a")

        async with StartupTransaction() as txn:
            txn.register_cleanup("a", cleanup_a)
            txn.mark_initialized("a")

        # No cleanups should have run
        assert cleanup_calls == []
        # Internal cleanup list should be cleared
        assert len(txn._cleanups) == 0


class TestStartupSLO:
    """Tests for SLO tracking."""

    @pytest.mark.asyncio
    async def test_slo_met(self):
        """Test SLO is met when startup is fast."""
        from aragora.server.startup_transaction import StartupTransaction

        async with StartupTransaction(slo_seconds=10) as txn:
            txn.mark_initialized("fast_component")

        report = txn.get_report()
        assert report.slo_met is True
        assert report.total_duration_seconds < report.slo_seconds

    @pytest.mark.asyncio
    async def test_elapsed_time_tracking(self):
        """Test elapsed time is tracked correctly."""
        from aragora.server.startup_transaction import StartupTransaction

        async with StartupTransaction() as txn:
            initial = txn.elapsed_seconds
            await asyncio.sleep(0.1)
            after_sleep = txn.elapsed_seconds

            assert after_sleep > initial
            assert after_sleep >= 0.1


class TestCheckpoints:
    """Tests for checkpoint creation."""

    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """Test creating checkpoints during startup."""
        from aragora.server.startup_transaction import StartupTransaction

        async with StartupTransaction() as txn:
            txn.mark_initialized("component_a")
            cp1 = txn.checkpoint("phase1")

            txn.mark_initialized("component_b")
            cp2 = txn.checkpoint("phase2")

        assert cp1.name == "phase1"
        assert cp2.name == "phase2"
        assert cp2.elapsed_seconds >= cp1.elapsed_seconds
        assert "component_a" in cp1.components
        assert "component_b" in cp2.components

    @pytest.mark.asyncio
    async def test_checkpoint_in_report(self):
        """Test checkpoints appear in report."""
        from aragora.server.startup_transaction import StartupTransaction

        async with StartupTransaction() as txn:
            txn.checkpoint("init")
            txn.checkpoint("done")

        report = txn.get_report()
        assert len(report.checkpoints) == 2
        assert report.checkpoints[0].name == "init"
        assert report.checkpoints[1].name == "done"


class TestRunStep:
    """Tests for run_step helper."""

    @pytest.mark.asyncio
    async def test_run_step_success(self):
        """Test run_step on successful initialization."""
        from aragora.server.startup_transaction import StartupTransaction

        init_called = False

        async def init_component():
            nonlocal init_called
            init_called = True
            return "initialized"

        async with StartupTransaction() as txn:
            result = await txn.run_step("component", init_component)

        assert init_called
        assert result == "initialized"
        assert "component" in txn._components_initialized

    @pytest.mark.asyncio
    async def test_run_step_failure_marks_failed(self):
        """Test run_step marks component as failed on exception."""
        from aragora.server.startup_transaction import StartupTransaction

        async def failing_init():
            raise ValueError("Init failed")

        try:
            async with StartupTransaction() as txn:
                await txn.run_step("bad_component", failing_init)
        except ValueError:
            pass

        assert "bad_component" in txn._components_failed

    @pytest.mark.asyncio
    async def test_run_step_with_cleanup(self):
        """Test run_step registers cleanup function."""
        from aragora.server.startup_transaction import StartupTransaction

        cleanup_called = False

        async def init_component():
            return True

        def cleanup_component():
            nonlocal cleanup_called
            cleanup_called = True

        try:
            async with StartupTransaction() as txn:
                await txn.run_step("component", init_component, cleanup_component)
                raise RuntimeError("Failure after init")
        except RuntimeError:
            pass

        assert cleanup_called


class TestStartupReport:
    """Tests for StartupReport."""

    @pytest.mark.asyncio
    async def test_report_to_dict(self):
        """Test report serialization to dictionary."""
        from aragora.server.startup_transaction import StartupTransaction

        async with StartupTransaction(slo_seconds=30) as txn:
            txn.mark_initialized("a")
            txn.checkpoint("cp1")

        report = txn.get_report()
        d = report.to_dict()

        assert "success" in d
        assert "total_duration_seconds" in d
        assert "slo_seconds" in d
        assert "slo_met" in d
        assert "components_initialized" in d
        assert "checkpoints" in d
        assert d["success"] is True
        assert d["slo_seconds"] == 30

    @pytest.mark.asyncio
    async def test_report_with_error(self):
        """Test report includes error message."""
        from aragora.server.startup_transaction import StartupTransaction

        try:
            async with StartupTransaction() as txn:
                raise ValueError("Test error")
        except ValueError:
            pass

        report = txn.get_report(error="Test error")
        assert report.error == "Test error"
        assert report.success is False


class TestGlobalReport:
    """Tests for global startup report functions."""

    def test_get_set_last_report(self):
        """Test getting and setting last startup report."""
        from aragora.server.startup_transaction import (
            StartupReport,
            get_last_startup_report,
            set_last_startup_report,
        )

        report = StartupReport(
            success=True,
            total_duration_seconds=5.0,
            slo_seconds=30.0,
            slo_met=True,
            components_initialized=10,
            components_failed=[],
            checkpoints=[],
        )

        set_last_startup_report(report)
        retrieved = get_last_startup_report()

        assert retrieved is report
        assert retrieved.success is True
        assert retrieved.components_initialized == 10
