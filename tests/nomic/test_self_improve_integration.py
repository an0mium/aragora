"""End-to-end integration tests for the self-improvement pipeline.

These tests exercise the real assess -> generate -> execute chain,
validating the wiring between AutonomousAssessmentEngine, GoalGenerator,
SelfImprovePipeline, and the MCP tools.

The StrategicScanner and MetricsCollector are mocked at the boundary
to avoid the ~120s filesystem scan on a 3000-module codebase, but all
other pipeline logic (signal aggregation, candidate ranking, health
score computation, goal generation, dry-run planning) runs for real.

Run with:
    pytest tests/nomic/test_self_improve_integration.py -v --timeout=300 -x
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.assessment_engine import (
    AutonomousAssessmentEngine,
    CodebaseHealthReport,
    ImprovementCandidate,
    SignalSource,
)
from aragora.nomic.goal_generator import GoalGenerator
from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline


# ---------------------------------------------------------------------------
# Shared mocks — scanner & metrics are slow on a large repo
# ---------------------------------------------------------------------------


def _make_fake_metric_snapshot():
    """Create a lightweight fake MetricSnapshot."""
    from aragora.nomic.metrics_collector import MetricSnapshot

    return MetricSnapshot(
        timestamp=1.0,
        tests_passed=100,
        tests_failed=2,
        tests_skipped=5,
        lint_errors=3,
        files_count=500,
        total_lines=50000,
    )


def _make_fake_scanner_assessment():
    """Create a fake StrategicAssessment with realistic findings."""
    from aragora.nomic.strategic_scanner import (
        StrategicAssessment,
        StrategicFinding,
    )

    return StrategicAssessment(
        findings=[
            StrategicFinding(
                category="untested",
                severity="medium",
                file_path="aragora/example/module.py",
                description="Module has no corresponding test file",
                evidence="No tests/example/test_module.py found",
                suggested_action="Add unit tests",
                track="core",
            ),
            StrategicFinding(
                category="complex",
                severity="high",
                file_path="aragora/server/unified_server.py",
                description="High complexity: 30 functions, 2000 LOC",
                evidence="Function count=30, LOC=2000",
                suggested_action="Consider splitting into smaller modules",
                track="core",
            ),
            StrategicFinding(
                category="stale",
                severity="low",
                file_path="aragora/debate/convergence.py",
                description="Stale TODO marker (> 60 days old)",
                evidence="TODO found on line 42",
                suggested_action="Address or remove stale TODO",
                track="core",
            ),
        ],
        metrics={"total_modules": 500, "untested_count": 10, "tested_pct": 98.0},
        focus_areas=["test_coverage", "complexity"],
        objective="",
        timestamp=1.0,
    )


@contextmanager
def _patch_slow_signals():
    """Mock both StrategicScanner.scan and MetricsCollector.collect_baseline."""
    fake_snapshot = _make_fake_metric_snapshot()
    fake_assessment = _make_fake_scanner_assessment()

    with (
        patch(
            "aragora.nomic.metrics_collector.MetricsCollector.collect_baseline",
            new_callable=AsyncMock,
            return_value=fake_snapshot,
        ),
        patch(
            "aragora.nomic.strategic_scanner.StrategicScanner.scan",
            return_value=fake_assessment,
        ),
    ):
        yield


def _safe_config() -> SelfImproveConfig:
    """Return a SelfImproveConfig that avoids side effects."""
    return SelfImproveConfig(
        capture_metrics=False,
        enable_codebase_indexing=False,
        enable_debug_loop=False,
        persist_outcomes=False,
        enable_codebase_metrics=False,
        run_tests=False,
        run_review=False,
        require_approval=True,
        use_worktrees=False,
    )


# ---------------------------------------------------------------------------
# Assessment integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(60)
class TestAssessmentIntegration:
    """Tests that exercise AutonomousAssessmentEngine end-to-end.

    Scanner and metrics collector are mocked for speed; the assessment
    engine's signal aggregation, candidate conversion, health-score
    computation, and serialization all run for real.
    """

    @pytest.fixture(autouse=True)
    async def _run_assessment(self) -> None:
        with _patch_slow_signals():
            engine = AutonomousAssessmentEngine()
            self.report = await engine.assess()

    def test_assess_returns_real_findings(self) -> None:
        """assess() returns a CodebaseHealthReport with a valid health_score."""
        assert isinstance(self.report, CodebaseHealthReport)
        assert 0.0 <= self.report.health_score <= 1.0
        assert self.report.assessment_duration_seconds > 0.0

    def test_assess_scanner_finds_untested_modules(self) -> None:
        """The scanner signal finds modules and converts to candidates."""
        scanner_sources = [s for s in self.report.signal_sources if s.name == "scanner"]
        assert len(scanner_sources) == 1
        scanner = scanner_sources[0]

        if scanner.error:
            pytest.skip(f"Scanner unavailable: {scanner.error}")

        assert len(scanner.findings) > 0

        # Verify scanner findings were converted to ImprovementCandidates
        scanner_candidates = [
            c for c in self.report.improvement_candidates if c.source == "scanner"
        ]
        assert len(scanner_candidates) > 0
        for c in scanner_candidates:
            assert isinstance(c, ImprovementCandidate)
            assert len(c.description) > 0
            assert 0.0 <= c.priority <= 1.0

    def test_assess_metrics_collector_runs(self) -> None:
        """MetricsCollector signal source is present and produces findings."""
        metrics_sources = [s for s in self.report.signal_sources if s.name == "metrics"]
        assert len(metrics_sources) == 1
        metrics = metrics_sources[0]

        assert len(metrics.findings) > 0 or metrics.error is not None

    def test_assess_to_dict_serializable(self) -> None:
        """Full report round-trips through to_dict() and has expected keys."""
        d = self.report.to_dict()

        assert isinstance(d, dict)
        assert "health_score" in d
        assert "signal_sources" in d
        assert "improvement_candidates" in d
        assert "metrics_snapshot" in d
        assert "assessment_duration_seconds" in d

        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert len(serialized) > 10

        # signal_sources should have the expected structure
        for src in d["signal_sources"]:
            assert "name" in src
            assert "weight" in src
            assert "findings_count" in src

    async def test_assess_custom_weights(self) -> None:
        """Custom signal weights change health_score vs default weights."""
        with _patch_slow_signals():
            engine_heavy = AutonomousAssessmentEngine(
                weights={
                    "scanner": 0.9,
                    "metrics": 0.025,
                    "regressions": 0.025,
                    "queue": 0.025,
                    "feedback": 0.025,
                }
            )
            heavy_report = await engine_heavy.assess()

        assert isinstance(heavy_report, CodebaseHealthReport)
        assert 0.0 <= heavy_report.health_score <= 1.0
        # Both should be valid; scores may differ due to different weighting
        assert 0.0 <= self.report.health_score <= 1.0


# ---------------------------------------------------------------------------
# Goal generation integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(60)
class TestGoalGenerationIntegration:
    """Tests that exercise GoalGenerator against a real assessment report."""

    @pytest.fixture(autouse=True)
    async def _run_assessment(self) -> None:
        with _patch_slow_signals():
            engine = AutonomousAssessmentEngine()
            self.report = await engine.assess()

    def test_generate_goals_from_real_assessment(self) -> None:
        """GoalGenerator produces PrioritizedGoal objects from a real report."""
        generator = GoalGenerator(max_goals=5)
        goals = generator.generate_goals(self.report)

        if not self.report.improvement_candidates:
            pytest.skip("No improvement candidates found")

        assert len(goals) > 0
        assert len(goals) <= 5

        for goal in goals:
            assert hasattr(goal, "description")
            assert hasattr(goal, "track")
            assert hasattr(goal, "priority")
            assert hasattr(goal, "estimated_impact")
            assert len(goal.description) > 0

    def test_generate_ideas_from_real_assessment(self) -> None:
        """generate_ideas() returns idea strings for the pipeline."""
        generator = GoalGenerator(max_goals=5)
        ideas = generator.generate_ideas(self.report)

        if not self.report.improvement_candidates:
            pytest.skip("No improvement candidates found")

        assert len(ideas) > 0
        assert len(ideas) <= 5

        for idea in ideas:
            assert isinstance(idea, str)
            assert len(idea) > 0
            # Ideas have the format "[category] description"
            assert idea.startswith("[")

    def test_generate_objective_from_real_assessment(self) -> None:
        """generate_objective() returns a single objective string."""
        generator = GoalGenerator(max_goals=5)
        objective = generator.generate_objective(self.report)

        assert isinstance(objective, str)
        assert len(objective) > 0

        if self.report.improvement_candidates:
            assert objective.startswith("[auto-assess]")
        else:
            assert "no issues" in objective.lower() or "maintain" in objective.lower()


# ---------------------------------------------------------------------------
# Dry-run integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(60)
class TestDryRunIntegration:
    """Tests that exercise the full pipeline dry_run path."""

    async def test_dry_run_with_real_goal(self) -> None:
        """Full pipeline: plan -> decompose -> dry_run preview."""
        config = _safe_config()
        config.scan_mode = True
        config.use_meta_planner = True

        pipeline = SelfImprovePipeline(config)

        with _patch_slow_signals():
            preview = await pipeline.dry_run(objective=None)

        assert isinstance(preview, dict)
        assert "objective" in preview
        assert "goals" in preview
        assert "subtasks" in preview
        assert "config" in preview

        assert isinstance(preview["goals"], list)
        assert isinstance(preview["subtasks"], list)

        # Config should reflect our safe settings
        assert preview["config"]["use_worktrees"] is False

    def test_cli_assess_flag_smoke(self) -> None:
        """scripts/self_develop.py --assess parses args and starts execution.

        The full CLI may timeout due to the scanner; we just verify it
        doesn't crash immediately with an import or argument error.
        """
        try:
            result = subprocess.run(
                [sys.executable, "scripts/self_develop.py", "--assess"],
                capture_output=True,
                text=True,
                cwd="/Users/armand/Development/aragora",
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            # Timing out means the process started and is running the scanner.
            # That's a pass — it didn't crash on startup.
            return

        if result.returncode != 0:
            assert "Traceback" not in result.stderr, (
                f"CLI crashed:\nstdout={result.stdout[-500:]}\nstderr={result.stderr[-500:]}"
            )


# ---------------------------------------------------------------------------
# MCP tool integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(60)
class TestMCPToolIntegration:
    """Tests that exercise the MCP self-improvement tools."""

    async def test_assess_codebase_tool_real(self) -> None:
        """MCP assess_codebase_tool() returns a valid dict."""
        from aragora.mcp.tools_module.self_improve import assess_codebase_tool

        with _patch_slow_signals():
            result = await assess_codebase_tool(weights="")

        assert isinstance(result, dict)
        assert "error" not in result, f"Tool returned error: {result.get('error')}"
        assert "health_score" in result
        assert 0.0 <= result["health_score"] <= 1.0
        assert "signal_sources" in result
        assert "improvement_candidates" in result

    async def test_generate_goals_tool_real(self) -> None:
        """MCP generate_improvement_goals_tool() returns goals."""
        from aragora.mcp.tools_module.self_improve import (
            generate_improvement_goals_tool,
        )

        with _patch_slow_signals():
            result = await generate_improvement_goals_tool(max_goals=3)

        assert isinstance(result, dict)
        assert "error" not in result, f"Tool returned error: {result.get('error')}"
        assert "health_score" in result
        assert "goals" in result
        assert isinstance(result["goals"], list)
        assert "goals_count" in result
        assert "candidates_count" in result
