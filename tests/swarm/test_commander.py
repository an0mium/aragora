"""Tests for SwarmCommander."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from aragora.swarm.commander import SwarmCommander, _ErrorResult
from aragora.swarm.config import SwarmCommanderConfig
from aragora.swarm.spec import SwarmSpec


@dataclass
class MockOrchestrationResult:
    """Mock of OrchestrationResult for testing."""

    total_subtasks: int = 3
    completed_subtasks: int = 2
    failed_subtasks: int = 1
    skipped_subtasks: int = 0
    assignments: list[Any] = field(default_factory=list)
    total_cost_usd: float = 1.50


class TestSwarmCommanderRunFromSpec:
    """Test SwarmCommander.run_from_spec."""

    @pytest.mark.asyncio
    async def test_run_from_spec_produces_report(self):
        """run_from_spec should produce a valid SwarmReport."""
        spec = SwarmSpec(
            raw_goal="Test goal",
            refined_goal="Test goal refined",
            budget_limit_usd=5.0,
        )

        mock_result = MockOrchestrationResult()
        output: list[str] = []

        commander = SwarmCommander()

        with patch.object(
            commander,
            "_dispatch",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            report = await commander.run_from_spec(spec, print_fn=lambda x: output.append(str(x)))

        assert report is not None
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    @pytest.mark.asyncio
    async def test_run_from_spec_sets_spec(self):
        """After run_from_spec, the spec property should be set."""
        spec = SwarmSpec(raw_goal="Check spec property")
        commander = SwarmCommander()

        with patch.object(
            commander,
            "_dispatch",
            new_callable=AsyncMock,
            return_value=MockOrchestrationResult(),
        ):
            await commander.run_from_spec(spec, print_fn=lambda _: None)

        assert commander.spec is spec


class TestSwarmCommanderDryRun:
    """Test SwarmCommander.dry_run."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_spec(self):
        """dry_run should return a SwarmSpec without dispatching."""
        output: list[str] = []
        commander = SwarmCommander()

        mock_spec = SwarmSpec(raw_goal="Dry run test", refined_goal="Dry run test")

        with patch.object(
            commander._interrogator,
            "interrogate",
            new_callable=AsyncMock,
            return_value=mock_spec,
        ):
            spec = await commander.dry_run(
                "Dry run test",
                input_fn=lambda _: "yes",
                print_fn=lambda x: output.append(str(x)),
            )

        assert spec.raw_goal == "Dry run test"
        # Should show spec JSON in output
        assert any("SPEC" in line for line in output)


class TestSwarmCommanderBuildOrchestrator:
    """Test orchestrator configuration from spec."""

    def test_build_orchestrator_passes_budget(self):
        """Budget limit from spec should flow to orchestrator."""
        spec = SwarmSpec(budget_limit_usd=15.0)
        commander = SwarmCommander()

        with patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as MockOrch:
            commander._build_orchestrator(spec)
            MockOrch.assert_called_once()
            call_kwargs = MockOrch.call_args[1]
            assert call_kwargs["budget_limit_usd"] == 15.0

    def test_build_orchestrator_passes_approval(self):
        """requires_approval from spec should flow to orchestrator."""
        spec = SwarmSpec(requires_approval=True)
        commander = SwarmCommander()

        with patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as MockOrch:
            commander._build_orchestrator(spec)
            call_kwargs = MockOrch.call_args[1]
            assert call_kwargs["require_human_approval"] is True


class TestErrorResult:
    """Test the _ErrorResult fallback object."""

    def test_error_result_attributes(self):
        err = _ErrorResult("something broke")
        assert err.error == "something broke"
        assert err.total_subtasks == 0
        assert err.failed_subtasks == 1
        assert err.completed_subtasks == 0
        assert err.total_cost_usd == 0.0

    def test_error_result_assignments_empty(self):
        err = _ErrorResult("error")
        assert err.assignments == []


class TestSwarmCommanderIterative:
    """Test iterative loop (run -> report -> 'what next?' -> repeat)."""

    @pytest.mark.asyncio
    async def test_iterative_loop_exits_on_done(self):
        """Typing 'done' exits the loop with one report."""
        commander = SwarmCommander()
        output: list[str] = []

        with patch.object(
            commander,
            "run",
            new_callable=AsyncMock,
            return_value=MagicMock(summary="Cycle 1 done"),
        ):
            reports = await commander.run_iterative(
                "Test goal",
                input_fn=lambda _: "done",
                print_fn=lambda x: output.append(str(x)),
            )

        assert len(reports) == 1

    @pytest.mark.asyncio
    async def test_iterative_loop_multiple_cycles(self):
        """Multiple goals run multiple cycles before exit."""
        commander = SwarmCommander()
        inputs = iter(["fix the dashboard", "done"])

        with patch.object(
            commander,
            "run",
            new_callable=AsyncMock,
            return_value=MagicMock(summary="Cycle done"),
        ):
            reports = await commander.run_iterative(
                "First goal",
                input_fn=lambda _: next(inputs),
                print_fn=lambda _: None,
            )

        assert len(reports) == 2

    @pytest.mark.asyncio
    async def test_iterative_disabled_runs_once(self):
        """When iterative_mode=False, runs once and exits."""
        config = SwarmCommanderConfig(iterative_mode=False)
        commander = SwarmCommander(config=config)

        with patch.object(
            commander,
            "run",
            new_callable=AsyncMock,
            return_value=MagicMock(summary="Single run"),
        ):
            reports = await commander.run_iterative(
                "One-shot goal",
                input_fn=lambda _: "should not be called",
                print_fn=lambda _: None,
            )

        assert len(reports) == 1


class TestSwarmCommanderConfig:
    """Test CEO-mode config defaults."""

    def test_default_budget_is_50(self):
        config = SwarmCommanderConfig()
        assert config.budget_limit_usd == 50.0

    def test_default_max_parallel_is_20(self):
        config = SwarmCommanderConfig()
        assert config.max_parallel_tasks == 20

    def test_default_iterative_mode_is_true(self):
        config = SwarmCommanderConfig()
        assert config.iterative_mode is True

    def test_max_parallel_flows_to_orchestrator(self):
        config = SwarmCommanderConfig(max_parallel_tasks=10)
        commander = SwarmCommander(config=config)
        spec = SwarmSpec(raw_goal="test")

        with patch("aragora.nomic.hardened_orchestrator.HardenedOrchestrator") as MockOrch:
            commander._build_orchestrator(spec)
            call_kwargs = MockOrch.call_args[1]
            assert call_kwargs["max_parallel_tasks"] == 10


class TestProactiveSuggestions:
    """Test proactive_suggestions field on SwarmSpec."""

    def test_spec_has_proactive_suggestions_field(self):
        spec = SwarmSpec(proactive_suggestions=["Add dark mode", "Fix mobile layout"])
        assert len(spec.proactive_suggestions) == 2
        assert "Add dark mode" in spec.proactive_suggestions

    def test_proactive_suggestions_in_dict(self):
        spec = SwarmSpec(proactive_suggestions=["suggestion 1"])
        data = spec.to_dict()
        assert "proactive_suggestions" in data
        assert data["proactive_suggestions"] == ["suggestion 1"]
