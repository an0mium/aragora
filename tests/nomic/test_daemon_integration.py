"""Integration tests for daemon skip-when-healthy / execute-when-unhealthy logic."""

import asyncio

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aragora.nomic.daemon import (
    CycleResult,
    DaemonConfig,
    DaemonState,
    DaemonStatus,
    SelfImprovementDaemon,
)


def _make_report(health_score=0.5, candidates=None):
    """Create a realistic CodebaseHealthReport-like object."""
    if candidates is None:
        candidates = [
            SimpleNamespace(
                description="Fix test coverage gap",
                priority=0.8,
                source="scanner",
                files=[],
                category="test",
            ),
            SimpleNamespace(
                description="Reduce complexity",
                priority=0.5,
                source="metrics",
                files=[],
                category="complexity",
            ),
        ]
    return SimpleNamespace(
        health_score=health_score,
        signal_sources=[],
        improvement_candidates=candidates,
        metrics_snapshot={},
        assessment_duration_seconds=0.5,
        to_dict=lambda: {"health_score": health_score},
    )


class TestDaemonDecisionLogic:
    """Test the daemon's skip/execute decision paths."""

    @pytest.mark.asyncio
    async def test_skips_when_health_above_threshold(self):
        """Cycle skipped when health_score >= threshold."""
        config = DaemonConfig(health_threshold=0.9, dry_run=True)
        daemon = SelfImprovementDaemon(config)

        with patch.object(
            daemon,
            "_assess",
            new_callable=AsyncMock,
            return_value=_make_report(health_score=0.95),
        ):
            result = await daemon.trigger_cycle()

        assert result.skipped is True
        assert "0.95" in result.skip_reason or "threshold" in result.skip_reason.lower()
        assert result.goal_executed == ""

    @pytest.mark.asyncio
    async def test_skips_when_too_few_candidates(self):
        """Cycle skipped when candidates < min_candidates."""
        config = DaemonConfig(health_threshold=0.9, min_candidates=5)
        daemon = SelfImprovementDaemon(config)

        report = _make_report(health_score=0.5, candidates=[
            SimpleNamespace(description="one", priority=0.5, source="s", files=[], category="c"),
        ])
        with patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=report):
            result = await daemon.trigger_cycle()

        assert result.skipped is True
        assert "candidates" in result.skip_reason.lower() or "1" in result.skip_reason

    @pytest.mark.asyncio
    async def test_executes_in_dry_run_mode(self):
        """Dry-run: passes threshold check, generates goals, but skips actual execution."""
        config = DaemonConfig(health_threshold=0.9, dry_run=True)
        daemon = SelfImprovementDaemon(config)

        report = _make_report(health_score=0.6)
        with patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=report):
            with patch.object(
                daemon,
                "_generate_goals",
                return_value=[SimpleNamespace(description="Improve coverage", priority=0.8)],
            ):
                result = await daemon.trigger_cycle()

        assert result.skipped is True
        assert result.skip_reason == "Dry run mode"
        assert result.success is True
        assert result.goal_executed == "Improve coverage"

    @pytest.mark.asyncio
    async def test_full_execution_captures_health_delta(self):
        """Non-dry-run cycle: assess -> execute -> re-assess records health_delta."""
        config = DaemonConfig(
            health_threshold=0.9,
            dry_run=False,
            require_approval=False,
            autonomous=True,
        )
        daemon = SelfImprovementDaemon(config)

        report_before = _make_report(health_score=0.6)
        report_after = _make_report(health_score=0.72)
        assess_returns = [report_before, report_after]

        async def mock_assess():
            return assess_returns.pop(0)

        pipeline_result = SimpleNamespace(
            subtasks_completed=2,
            subtasks_failed=0,
            regressions_detected=False,
            to_dict=lambda: {},
        )

        with patch.object(daemon, "_assess", side_effect=mock_assess):
            with patch.object(
                daemon,
                "_generate_goals",
                return_value=[SimpleNamespace(description="Reduce tech debt")],
            ):
                with patch.object(
                    daemon, "_execute", new_callable=AsyncMock, return_value=pipeline_result
                ):
                    with patch.object(daemon, "_record_outcome"):
                        result = await daemon.trigger_cycle()

        assert result.health_before == pytest.approx(0.6)
        assert result.health_after == pytest.approx(0.72)
        assert result.health_delta == pytest.approx(0.12)
        assert result.success is True
        assert result.skipped is False
        assert result.goal_executed == "Reduce tech debt"

    @pytest.mark.asyncio
    async def test_consecutive_failures_stop_daemon_loop(self):
        """Loop exits after max_consecutive_failures are reached."""
        config = DaemonConfig(
            max_consecutive_failures=2,
            max_cycles=10,
            interval_seconds=0.01,
            cooldown_after_failure_seconds=0.01,
            health_threshold=1.0,
        )
        daemon = SelfImprovementDaemon(config)

        async def failing_assess():
            raise RuntimeError("Assessment engine unavailable")

        with patch.object(daemon, "_assess", side_effect=failing_assess):
            await daemon.start()
            # Wait enough for 2 failing cycles + cooldowns
            await asyncio.sleep(1.0)
            await daemon.stop()

        status = daemon.get_status()
        assert status.consecutive_failures >= 2
        assert daemon.state == DaemonState.STOPPED

    @pytest.mark.asyncio
    async def test_cumulative_budget_stops_daemon_loop(self):
        """Loop exits when cumulative budget is exhausted."""
        config = DaemonConfig(
            budget_limit_cumulative_usd=1.0,
            max_cycles=10,
            interval_seconds=0.01,
            health_threshold=1.0,
        )
        daemon = SelfImprovementDaemon(config)
        # Pre-exhaust the budget
        daemon._cumulative_budget_usd = 100.0

        await daemon.start()
        await asyncio.sleep(0.3)
        await daemon.stop()

        assert daemon.state == DaemonState.STOPPED
        # Loop should have exited before running any cycle
        status = daemon.get_status()
        assert status.cycles_completed == 0

    @pytest.mark.asyncio
    async def test_daemon_status_reflects_idle_state(self):
        """get_status() returns correct DaemonStatus for a fresh daemon."""
        config = DaemonConfig()
        daemon = SelfImprovementDaemon(config)

        status = daemon.get_status()
        assert isinstance(status, DaemonStatus)
        assert status.state == "idle"
        assert status.cycles_completed == 0
        assert status.cycles_failed == 0
        assert status.consecutive_failures == 0
        assert status.last_health_score is None
        assert status.cumulative_budget_used_usd == 0.0
        assert status.history == []

        d = status.to_dict()
        assert isinstance(d, dict)
        assert d["state"] == "idle"
        assert "history" in d

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_start_stop_lifecycle_no_hang(self):
        """start() / stop() completes within timeout without blocking."""
        config = DaemonConfig(
            interval_seconds=100,
            max_cycles=1,
            health_threshold=0.01,  # Low threshold so cycle skips quickly
        )
        daemon = SelfImprovementDaemon(config)

        with patch.object(
            daemon,
            "_assess",
            new_callable=AsyncMock,
            return_value=_make_report(health_score=0.99),
        ):
            await daemon.start()
            assert daemon.state in (
                DaemonState.RUNNING,
                DaemonState.ASSESSING,
                DaemonState.STOPPED,
            )
            # Give the loop time to run the single cycle
            await asyncio.sleep(0.3)
            await daemon.stop()

        assert daemon.state == DaemonState.STOPPED
