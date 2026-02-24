"""Tests for SelfImprovementDaemon — continuous autonomous improvement loop."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_health_report(health_score: float = 0.7, candidates: list | None = None):
    """Build a mock CodebaseHealthReport."""
    return types.SimpleNamespace(
        health_score=health_score,
        improvement_candidates=candidates
        or [
            types.SimpleNamespace(
                description="Fix 3 failing tests",
                priority=0.9,
                source="scanner",
                files=["tests/test_foo.py"],
                category="test",
            ),
        ],
        metrics_snapshot={},
    )


def _mock_pipeline_result(completed: int = 1, failed: int = 0, regressions: bool = False):
    """Build a mock SelfImproveResult."""
    return types.SimpleNamespace(
        subtasks_completed=completed,
        subtasks_failed=failed,
        regressions_detected=regressions,
        objective="Fix tests",
    )


def _mock_goal(desc: str = "Fix 3 failing tests"):
    """Build a mock PrioritizedGoal."""
    return types.SimpleNamespace(description=desc)


class TestDaemonConfig:
    """Tests for DaemonConfig defaults."""

    def test_default_config(self):
        from aragora.nomic.daemon import DaemonConfig

        config = DaemonConfig()
        assert config.health_threshold == 0.95
        assert config.dry_run is False
        assert config.max_consecutive_failures == 3
        assert config.interval_seconds == 3600.0

    def test_custom_config(self):
        from aragora.nomic.daemon import DaemonConfig

        config = DaemonConfig(
            health_threshold=0.8,
            dry_run=True,
            max_cycles=5,
        )
        assert config.health_threshold == 0.8
        assert config.dry_run is True
        assert config.max_cycles == 5


class TestDaemonState:
    """Tests for daemon state management."""

    def test_initial_state_is_idle(self):
        from aragora.nomic.daemon import DaemonState, SelfImprovementDaemon

        daemon = SelfImprovementDaemon()
        assert daemon.state == DaemonState.IDLE

    def test_get_status_when_idle(self):
        from aragora.nomic.daemon import SelfImprovementDaemon

        daemon = SelfImprovementDaemon()
        status = daemon.get_status()
        assert status.state == "idle"
        assert status.cycles_completed == 0
        assert status.cycles_failed == 0

    def test_status_to_dict(self):
        from aragora.nomic.daemon import SelfImprovementDaemon

        daemon = SelfImprovementDaemon()
        d = daemon.get_status().to_dict()
        assert "state" in d
        assert "cycles_completed" in d
        assert "history" in d


class TestCycleResult:
    """Tests for CycleResult."""

    def test_cycle_result_to_dict(self):
        from aragora.nomic.daemon import CycleResult

        result = CycleResult(
            cycle_number=1,
            health_before=0.7,
            health_after=0.75,
            health_delta=0.05,
            goal_executed="Fix tests",
            success=True,
        )
        d = result.to_dict()
        assert d["cycle_number"] == 1
        assert d["health_before"] == 0.7
        assert d["success"] is True

    def test_cycle_result_skipped(self):
        from aragora.nomic.daemon import CycleResult

        result = CycleResult(
            cycle_number=1,
            health_before=0.98,
            skipped=True,
            skip_reason="Health above threshold",
        )
        assert result.skipped is True
        assert "threshold" in result.skip_reason


class TestTriggerCycle:
    """Tests for manual cycle triggering."""

    @pytest.mark.asyncio
    async def test_trigger_cycle_skips_when_healthy(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(health_threshold=0.9)
        daemon = SelfImprovementDaemon(config)

        healthy_report = _mock_health_report(health_score=0.95)

        with patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=healthy_report):
            result = await daemon.trigger_cycle()

        assert result.skipped is True
        assert "threshold" in result.skip_reason

    @pytest.mark.asyncio
    async def test_trigger_cycle_executes_when_unhealthy(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            health_threshold=0.9, dry_run=False, require_approval=False, autonomous=True
        )
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(health_score=0.6)
        after_report = _mock_health_report(health_score=0.65)
        pipeline_result = _mock_pipeline_result(completed=1)
        goals = [_mock_goal()]

        with (
            patch.object(
                daemon, "_assess", new_callable=AsyncMock, side_effect=[report, after_report]
            ),
            patch.object(daemon, "_generate_goals", return_value=goals),
            patch.object(daemon, "_execute", new_callable=AsyncMock, return_value=pipeline_result),
            patch.object(daemon, "_record_outcome"),
        ):
            result = await daemon.trigger_cycle()

        assert result.success is True
        assert result.health_before == 0.6
        assert result.health_after == 0.65
        assert result.health_delta == pytest.approx(0.05, abs=0.001)

    @pytest.mark.asyncio
    async def test_trigger_cycle_dry_run(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(dry_run=True)
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(health_score=0.6)
        goals = [_mock_goal()]

        with (
            patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=report),
            patch.object(daemon, "_generate_goals", return_value=goals),
        ):
            result = await daemon.trigger_cycle()

        assert result.skipped is True
        assert result.skip_reason == "Dry run mode"

    @pytest.mark.asyncio
    async def test_trigger_cycle_no_goals_generated(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig()
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(health_score=0.6)

        with (
            patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=report),
            patch.object(daemon, "_generate_goals", return_value=[]),
        ):
            result = await daemon.trigger_cycle()

        assert result.skipped is True
        assert "No goals" in result.skip_reason

    @pytest.mark.asyncio
    async def test_trigger_cycle_handles_error(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig()
        daemon = SelfImprovementDaemon(config)

        with patch.object(
            daemon, "_assess", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ):
            result = await daemon.trigger_cycle()

        assert result.error is not None
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_trigger_cycle_skips_few_candidates(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(min_candidates=5)
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(
            health_score=0.6,
            candidates=[
                types.SimpleNamespace(
                    description="One issue", priority=0.8, source="s", files=[], category="test"
                ),
            ],
        )

        with patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=report):
            result = await daemon.trigger_cycle()

        assert result.skipped is True
        assert "candidates" in result.skip_reason


class TestDaemonLoop:
    """Tests for the continuous loop."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        from aragora.nomic.daemon import DaemonConfig, DaemonState, SelfImprovementDaemon

        config = DaemonConfig(
            interval_seconds=0.01,
            max_cycles=1,
            health_threshold=0.0,  # Never skip
        )
        daemon = SelfImprovementDaemon(config)

        healthy_report = _mock_health_report(health_score=0.99)

        with patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=healthy_report):
            await daemon.start()
            # Wait a bit for the loop to complete its max_cycles
            await asyncio.sleep(0.1)

        # Loop should have exited after max_cycles
        status = daemon.get_status()
        assert status.state == "stopped"

    @pytest.mark.asyncio
    async def test_stop_interrupts_sleep(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            interval_seconds=9999,  # Very long sleep
            max_cycles=0,  # Unlimited
        )
        daemon = SelfImprovementDaemon(config)

        healthy_report = _mock_health_report(health_score=0.99)

        with patch.object(daemon, "_assess", new_callable=AsyncMock, return_value=healthy_report):
            await daemon.start()
            await asyncio.sleep(0.05)  # Let first cycle run (it will skip)
            await daemon.stop()

        status = daemon.get_status()
        assert status.state == "stopped"

    @pytest.mark.asyncio
    async def test_consecutive_failures_stop_daemon(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            max_consecutive_failures=2,
            interval_seconds=0.01,
            cooldown_after_failure_seconds=0.01,
        )
        daemon = SelfImprovementDaemon(config)

        with patch.object(
            daemon, "_assess", new_callable=AsyncMock, side_effect=RuntimeError("fail")
        ):
            await daemon.start()
            await asyncio.sleep(0.2)

        status = daemon.get_status()
        assert status.state == "stopped"
        assert status.consecutive_failures >= 2

    @pytest.mark.asyncio
    async def test_budget_limit_stops_daemon(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            budget_limit_cumulative_usd=0.0,  # Already exhausted
            interval_seconds=0.01,
        )
        daemon = SelfImprovementDaemon(config)

        await daemon.start()
        await asyncio.sleep(0.1)

        status = daemon.get_status()
        assert status.state == "stopped"


class TestDaemonRecordOutcome:
    """Tests for outcome recording."""

    def test_record_outcome_calls_meta_planner(self):
        from aragora.nomic.daemon import SelfImprovementDaemon

        daemon = SelfImprovementDaemon()
        pipeline_result = _mock_pipeline_result(completed=2, failed=0)

        with patch("aragora.nomic.meta_planner.MetaPlanner") as mock_cls:
            mock_planner = MagicMock()
            mock_cls.return_value = mock_planner

            daemon._record_outcome("Fix tests", pipeline_result)

            mock_planner.record_outcome.assert_called_once()
            call_args = mock_planner.record_outcome.call_args
            assert call_args.kwargs["objective"] == "Fix tests"

    def test_record_outcome_graceful_on_import_error(self):
        from aragora.nomic.daemon import SelfImprovementDaemon

        daemon = SelfImprovementDaemon()
        pipeline_result = _mock_pipeline_result()

        with patch.dict("sys.modules", {"aragora.nomic.meta_planner": None}):
            # Should not raise
            daemon._record_outcome("Fix tests", pipeline_result)


class TestAutoExecuteLowRisk:
    """Tests for auto-execute low-risk goals feature."""

    def test_config_defaults(self):
        from aragora.nomic.daemon import DaemonConfig

        config = DaemonConfig()
        assert config.auto_execute_low_risk is False
        assert config.low_risk_threshold == 0.3

    def test_config_custom(self):
        from aragora.nomic.daemon import DaemonConfig

        config = DaemonConfig(auto_execute_low_risk=True, low_risk_threshold=0.2)
        assert config.auto_execute_low_risk is True
        assert config.low_risk_threshold == 0.2

    @pytest.mark.asyncio
    async def test_low_risk_goal_bypasses_approval(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            auto_execute_low_risk=True,
            low_risk_threshold=0.3,
            require_approval=True,
            dry_run=False,
        )
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(health_score=0.6)
        after_report = _mock_health_report(health_score=0.65)
        pipeline_result = _mock_pipeline_result(completed=1)

        # Goal with low risk score
        low_risk_goal = types.SimpleNamespace(
            description="Fix lint warnings", risk_score=0.1
        )

        execute_mock = AsyncMock(return_value=pipeline_result)

        with (
            patch.object(
                daemon, "_assess", new_callable=AsyncMock,
                side_effect=[report, after_report],
            ),
            patch.object(daemon, "_generate_goals", return_value=[low_risk_goal]),
            patch.object(daemon, "_execute", execute_mock),
            patch.object(daemon, "_record_outcome"),
        ):
            result = await daemon.trigger_cycle()

        assert result.success is True
        # Verify _execute was called with require_approval=False
        execute_mock.assert_called_once_with(
            "Fix lint warnings", require_approval=False
        )

    @pytest.mark.asyncio
    async def test_high_risk_goal_keeps_approval(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            auto_execute_low_risk=True,
            low_risk_threshold=0.3,
            require_approval=True,
            dry_run=False,
        )
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(health_score=0.6)
        after_report = _mock_health_report(health_score=0.65)
        pipeline_result = _mock_pipeline_result(completed=1)

        # Goal with high risk score
        high_risk_goal = types.SimpleNamespace(
            description="Refactor core orchestrator", risk_score=0.8
        )

        execute_mock = AsyncMock(return_value=pipeline_result)

        with (
            patch.object(
                daemon, "_assess", new_callable=AsyncMock,
                side_effect=[report, after_report],
            ),
            patch.object(daemon, "_generate_goals", return_value=[high_risk_goal]),
            patch.object(daemon, "_execute", execute_mock),
            patch.object(daemon, "_record_outcome"),
        ):
            result = await daemon.trigger_cycle()

        assert result.success is True
        # Verify _execute was called with require_approval=True (not bypassed)
        execute_mock.assert_called_once_with(
            "Refactor core orchestrator", require_approval=True
        )

    @pytest.mark.asyncio
    async def test_auto_execute_disabled_keeps_approval(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            auto_execute_low_risk=False,  # Disabled
            require_approval=True,
            dry_run=False,
        )
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(health_score=0.6)
        after_report = _mock_health_report(health_score=0.65)
        pipeline_result = _mock_pipeline_result(completed=1)

        low_risk_goal = types.SimpleNamespace(
            description="Fix typo", risk_score=0.05
        )

        execute_mock = AsyncMock(return_value=pipeline_result)

        with (
            patch.object(
                daemon, "_assess", new_callable=AsyncMock,
                side_effect=[report, after_report],
            ),
            patch.object(daemon, "_generate_goals", return_value=[low_risk_goal]),
            patch.object(daemon, "_execute", execute_mock),
            patch.object(daemon, "_record_outcome"),
        ):
            result = await daemon.trigger_cycle()

        # Even though goal is low risk, feature is disabled → approval kept
        execute_mock.assert_called_once_with(
            "Fix typo", require_approval=True
        )

    @pytest.mark.asyncio
    async def test_goal_without_risk_score_uses_default_high(self):
        from aragora.nomic.daemon import DaemonConfig, SelfImprovementDaemon

        config = DaemonConfig(
            auto_execute_low_risk=True,
            low_risk_threshold=0.3,
            require_approval=True,
            dry_run=False,
        )
        daemon = SelfImprovementDaemon(config)

        report = _mock_health_report(health_score=0.6)
        after_report = _mock_health_report(health_score=0.65)
        pipeline_result = _mock_pipeline_result(completed=1)

        # Goal without risk_score attribute
        goal_no_risk = types.SimpleNamespace(description="Unknown risk goal")

        execute_mock = AsyncMock(return_value=pipeline_result)

        with (
            patch.object(
                daemon, "_assess", new_callable=AsyncMock,
                side_effect=[report, after_report],
            ),
            patch.object(daemon, "_generate_goals", return_value=[goal_no_risk]),
            patch.object(daemon, "_execute", execute_mock),
            patch.object(daemon, "_record_outcome"),
        ):
            result = await daemon.trigger_cycle()

        # No risk_score → defaults to 1.0 → approval required
        execute_mock.assert_called_once_with(
            "Unknown risk goal", require_approval=True
        )


class TestStartupIntegration:
    """Tests for server startup daemon integration."""

    @pytest.mark.asyncio
    async def test_daemon_disabled_by_default(self):
        from aragora.server.startup.background import init_self_improvement_daemon

        with patch.dict("os.environ", {}, clear=False):
            # Remove the env var if present
            import os
            os.environ.pop("ARAGORA_SELF_IMPROVE_ENABLED", None)
            result = await init_self_improvement_daemon()
        assert result is None

    @pytest.mark.asyncio
    async def test_daemon_disabled_during_tests(self):
        from aragora.server.startup.background import init_self_improvement_daemon

        with patch.dict("os.environ", {
            "PYTEST_CURRENT_TEST": "test_foo.py",
            "ARAGORA_SELF_IMPROVE_ENABLED": "true",
        }):
            result = await init_self_improvement_daemon()
        assert result is None

    @pytest.mark.asyncio
    async def test_daemon_starts_when_enabled(self):
        from aragora.server.startup.background import init_self_improvement_daemon

        mock_daemon = MagicMock()
        mock_task = MagicMock()
        mock_daemon._task = mock_task
        mock_daemon.start = AsyncMock()

        with (
            patch.dict("os.environ", {
                "ARAGORA_SELF_IMPROVE_ENABLED": "true",
                "ARAGORA_SELF_IMPROVE_DRY_RUN": "true",
            }, clear=False),
            patch(
                "aragora.nomic.daemon.SelfImprovementDaemon",
                return_value=mock_daemon,
            ),
            patch(
                "aragora.server.startup.background._store_daemon_singleton",
            ),
        ):
            # Need to remove PYTEST_CURRENT_TEST entirely
            import os
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            result = await init_self_improvement_daemon()

        mock_daemon.start.assert_called_once()
        assert result == mock_task
