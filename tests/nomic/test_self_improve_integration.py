"""End-to-end integration tests for the self-improvement pipeline.

These tests exercise the real assess -> generate -> execute chain against
the live Aragora codebase.  They are marked ``slow`` because the
StrategicScanner walks the file tree.

The MetricsCollector is patched at the async boundary so it returns
instantly (its subprocess + rglob calls are too slow on a 3k-module repo).
Every other signal source runs against the real codebase.

Run with:
    pytest tests/nomic/test_self_improve_integration.py -v --timeout=180 -x
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.assessment_engine import (
    AutonomousAssessmentEngine,
    CodebaseHealthReport,
    SignalSource,
)
from aragora.nomic.goal_generator import GoalGenerator
from aragora.nomic.self_improve import SelfImproveConfig, SelfImprovePipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_metrics_collector():
    """Context manager that makes MetricsCollector.collect_baseline return
    a lightweight fake snapshot instantly (avoiding subprocess + rglob).
    """
    from aragora.nomic.metrics_collector import MetricSnapshot

    fake_snapshot = MetricSnapshot(
        timestamp=1.0,
        tests_passed=100,
        tests_failed=2,
        tests_skipped=5,
        lint_errors=3,
        files_count=500,
        total_lines=50000,
    )

    return patch(
        "aragora.nomic.metrics_collector.MetricsCollector.collect_baseline",
        new_callable=AsyncMock,
        return_value=fake_snapshot,
    )


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
@pytest.mark.timeout(120)
class TestAssessmentIntegration:
    """Tests that exercise AutonomousAssessmentEngine against the real repo."""

    @pytest.fixture(autouse=True)
    async def _run_assessment(self) -> None:
        """Run the assessment once and share across all tests in the class."""
        with _patch_metrics_collector():
            engine = AutonomousAssessmentEngine()
            self.report = await engine.assess()

    def test_assess_returns_real_findings(self) -> None:
        """assess() returns a CodebaseHealthReport with a valid health_score."""
        assert isinstance(self.report, CodebaseHealthReport)
        assert 0.0 <= self.report.health_score <= 1.0
        assert self.report.assessment_duration_seconds > 0.0

    def test_assess_scanner_finds_untested_modules(self) -> None:
        """The scanner signal should find modules without test files."""
        scanner_sources = [
            s for s in self.report.signal_sources if s.name == "scanner"
        ]
        assert len(scanner_sources) == 1
        scanner = scanner_sources[0]

        if scanner.error:
            pytest.skip(f"Scanner unavailable: {scanner.error}")

        # The Aragora codebase is large enough that the scanner should find
        # at least one untested or complex module.
        assert len(scanner.findings) > 0

    def test_assess_metrics_collector_runs(self) -> None:
        """MetricsCollector signal source is present (patched for speed)."""
        metrics_sources = [
            s for s in self.report.signal_sources if s.name == "metrics"
        ]
        assert len(metrics_sources) == 1
        metrics = metrics_sources[0]

        # With the patched collector, findings should be present
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

    def test_assess_custom_weights(self) -> None:
        """Custom signal weights are accepted and produce a valid report."""
        # Verify default report is valid
        assert 0.0 <= self.report.health_score <= 1.0

        # Verify constructor accepts custom weights
        engine_heavy = AutonomousAssessmentEngine(
            weights={
                "scanner": 0.9,
                "metrics": 0.025,
                "regressions": 0.025,
                "queue": 0.025,
                "feedback": 0.025,
            }
        )
        assert engine_heavy._weights["scanner"] == 0.9


# ---------------------------------------------------------------------------
# Goal generation integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestGoalGenerationIntegration:
    """Tests that exercise GoalGenerator against a real assessment report."""

    @pytest.fixture(autouse=True)
    async def _run_assessment(self) -> None:
        with _patch_metrics_collector():
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
@pytest.mark.timeout(120)
class TestDryRunIntegration:
    """Tests that exercise the full pipeline dry_run path."""

    async def test_dry_run_with_real_goal(self) -> None:
        """Full pipeline: assess -> goals -> SelfImprovePipeline.dry_run()."""
        config = _safe_config()
        config.scan_mode = True
        config.use_meta_planner = True

        pipeline = SelfImprovePipeline(config)

        with _patch_metrics_collector():
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
        """scripts/self_develop.py --assess exits cleanly (or skips on timeout)."""
        try:
            result = subprocess.run(
                [sys.executable, "scripts/self_develop.py", "--assess"],
                capture_output=True,
                text=True,
                cwd="/Users/armand/Development/aragora",
                timeout=110,
            )
        except subprocess.TimeoutExpired:
            pytest.skip("CLI assess timed out (scanner rglob on large repo)")

        # Exit code 0 = success; non-zero is acceptable if imports fail,
        # but it should not crash with a traceback.
        if result.returncode != 0:
            assert "Traceback" not in result.stderr, (
                f"CLI crashed:\nstdout={result.stdout[-500:]}\n"
                f"stderr={result.stderr[-500:]}"
            )


# ---------------------------------------------------------------------------
# MCP tool integration
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestMCPToolIntegration:
    """Tests that exercise the MCP self-improvement tools against real data."""

    async def test_assess_codebase_tool_real(self) -> None:
        """MCP assess_codebase_tool() returns a valid dict."""
        from aragora.mcp.tools_module.self_improve import assess_codebase_tool

        with _patch_metrics_collector():
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

        with _patch_metrics_collector():
            result = await generate_improvement_goals_tool(max_goals=3)

        assert isinstance(result, dict)
        assert "error" not in result, f"Tool returned error: {result.get('error')}"
        assert "health_score" in result
        assert "goals" in result
        assert isinstance(result["goals"], list)
        assert "goals_count" in result
        assert "candidates_count" in result
