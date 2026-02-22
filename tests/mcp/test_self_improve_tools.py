"""Tests for MCP self-improvement tools."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_health_report(health_score: float = 0.7, candidates: list | None = None):
    """Build a mock CodebaseHealthReport."""
    report = types.SimpleNamespace(
        health_score=health_score,
        improvement_candidates=candidates or [],
        metrics_snapshot={},
        signal_sources=[],
        assessment_duration_seconds=0.1,
    )
    report.to_dict = lambda: {
        "health_score": report.health_score,
        "signal_sources": [],
        "improvement_candidates": [],
        "metrics_snapshot": {},
        "assessment_duration_seconds": 0.1,
    }
    return report


class TestAssessCodebaseTool:
    """Tests for assess_codebase_tool."""

    @pytest.mark.asyncio
    async def test_assess_returns_health_report(self):
        report = _mock_health_report(health_score=0.8)

        with patch("aragora.nomic.assessment_engine.AutonomousAssessmentEngine") as mock_cls:
            mock_engine = MagicMock()
            mock_engine.assess = AsyncMock(return_value=report)
            mock_cls.return_value = mock_engine

            from aragora.mcp.tools_module.self_improve import assess_codebase_tool
            result = await assess_codebase_tool()

        assert result["health_score"] == 0.8

    @pytest.mark.asyncio
    async def test_assess_graceful_on_import_error(self):
        with patch.dict("sys.modules", {"aragora.nomic.assessment_engine": None}):
            from aragora.mcp.tools_module import self_improve
            # Force reimport to pick up the None module
            import importlib
            importlib.reload(self_improve)

            result = await self_improve.assess_codebase_tool()

        assert "error" in result

    @pytest.mark.asyncio
    async def test_assess_with_invalid_weights(self):
        from aragora.mcp.tools_module.self_improve import assess_codebase_tool
        result = await assess_codebase_tool(weights="not-json")
        assert "error" in result


class TestGenerateGoalsTool:
    """Tests for generate_improvement_goals_tool."""

    @pytest.mark.asyncio
    async def test_generate_goals_returns_goals(self):
        report = _mock_health_report(health_score=0.6)

        mock_goal = types.SimpleNamespace(
            id="auto_0_test",
            description="Fix tests",
            track=types.SimpleNamespace(value="qa"),
            estimated_impact="high",
            priority=1,
            file_hints=["tests/test_foo.py"],
        )

        with (
            patch("aragora.nomic.assessment_engine.AutonomousAssessmentEngine") as mock_engine_cls,
            patch("aragora.nomic.goal_generator.GoalGenerator") as mock_gen_cls,
        ):
            mock_engine = MagicMock()
            mock_engine.assess = AsyncMock(return_value=report)
            mock_engine_cls.return_value = mock_engine

            mock_gen = MagicMock()
            mock_gen.generate_goals.return_value = [mock_goal]
            mock_gen_cls.return_value = mock_gen

            from aragora.mcp.tools_module.self_improve import generate_improvement_goals_tool
            result = await generate_improvement_goals_tool(max_goals=3)

        assert result["goals_count"] == 1
        assert result["goals"][0]["description"] == "Fix tests"


class TestRunSelfImprovementTool:
    """Tests for run_self_improvement_tool."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_preview(self):
        mock_preview = {"objective": "test", "goals": [], "subtasks": []}

        with patch("aragora.nomic.self_improve.SelfImprovePipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.dry_run = AsyncMock(return_value=mock_preview)
            mock_cls.return_value = mock_pipeline

            from aragora.mcp.tools_module.self_improve import run_self_improvement_tool
            result = await run_self_improvement_tool(objective="Fix tests", dry_run=True)

        assert result["mode"] == "dry_run"
        assert "preview" in result


class TestGetDaemonStatusTool:
    """Tests for get_daemon_status_tool."""

    @pytest.mark.asyncio
    async def test_get_status_returns_dict(self):
        mock_status = types.SimpleNamespace(
            state="idle",
            cycles_completed=0,
            cycles_failed=0,
            consecutive_failures=0,
            last_health_score=None,
            last_cycle_time=None,
            cumulative_budget_used_usd=0.0,
            history=[],
        )
        mock_status.to_dict = lambda: {
            "state": "idle",
            "cycles_completed": 0,
            "cycles_failed": 0,
            "consecutive_failures": 0,
            "last_health_score": None,
            "last_cycle_time": None,
            "cumulative_budget_used_usd": 0.0,
            "history": [],
        }

        with patch("aragora.mcp.tools_module.self_improve._get_daemon") as mock_get:
            mock_daemon = MagicMock()
            mock_daemon.get_status.return_value = mock_status
            mock_get.return_value = mock_daemon

            from aragora.mcp.tools_module.self_improve import get_daemon_status_tool
            result = await get_daemon_status_tool()

        assert result["state"] == "idle"


class TestTriggerCycleTool:
    """Tests for trigger_improvement_cycle_tool."""

    @pytest.mark.asyncio
    async def test_trigger_cycle_dry_run(self):
        mock_result = types.SimpleNamespace(
            cycle_number=1,
            health_before=0.7,
            health_after=None,
            health_delta=0.0,
            goal_executed="",
            success=True,
            skipped=True,
            skip_reason="Dry run mode",
            duration_seconds=0.1,
            error=None,
        )
        mock_result.to_dict = lambda: {
            "cycle_number": 1,
            "health_before": 0.7,
            "health_after": None,
            "health_delta": 0.0,
            "goal_executed": "",
            "success": True,
            "skipped": True,
            "skip_reason": "Dry run mode",
            "duration_seconds": 0.1,
            "error": None,
        }

        with patch("aragora.nomic.daemon.SelfImprovementDaemon") as mock_cls:
            mock_daemon = MagicMock()
            mock_daemon.trigger_cycle = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_daemon

            from aragora.mcp.tools_module.self_improve import trigger_improvement_cycle_tool
            result = await trigger_improvement_cycle_tool(dry_run=True)

        assert result["skipped"] is True
        assert result["skip_reason"] == "Dry run mode"
