"""Tests for cross-test impact analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.nomic.testfixer.impact import ImpactAnalyzer, ImpactResult


@dataclass
class FakeFailure:
    test_name: str
    test_file: str = "tests/test_foo.py"
    error_type: str = "AssertionError"
    error_message: str = "failed"
    stack_trace: str = ""


@dataclass
class FakeTestResult:
    failures: list = field(default_factory=list)
    success: bool = False
    total_tests: int = 10
    passed: int = 8


class TestImpactResult:
    def test_summary_with_new_failures(self):
        result = ImpactResult(
            new_failures=[FakeFailure("test_new")],
            resolved_failures=["test_old"],
            unchanged_failures=["test_same"],
            has_regressions=True,
        )
        summary = result.summary
        assert "1 new" in summary
        assert "1 resolved" in summary
        assert "1 unchanged" in summary

    def test_summary_no_changes(self):
        result = ImpactResult()
        assert result.summary == "no changes"


class TestImpactAnalyzer:
    @pytest.mark.asyncio
    async def test_detects_regressions(self):
        runner = MagicMock()
        runner.run_full = AsyncMock(
            return_value=FakeTestResult(
                failures=[
                    FakeFailure("test_a"),  # was already failing
                    FakeFailure("test_NEW"),  # new regression
                ],
            )
        )

        analyzer = ImpactAnalyzer(runner)
        result = await analyzer.check_impact(["test_a"])

        assert result.has_regressions is True
        assert len(result.new_failures) == 1
        assert result.new_failures[0].test_name == "test_NEW"
        assert result.resolved_failures == []
        assert result.unchanged_failures == ["test_a"]

    @pytest.mark.asyncio
    async def test_detects_resolved(self):
        runner = MagicMock()
        runner.run_full = AsyncMock(
            return_value=FakeTestResult(failures=[])
        )

        analyzer = ImpactAnalyzer(runner)
        result = await analyzer.check_impact(["test_was_broken"])

        assert result.has_regressions is False
        assert result.resolved_failures == ["test_was_broken"]
        assert result.new_failures == []

    @pytest.mark.asyncio
    async def test_no_change(self):
        runner = MagicMock()
        runner.run_full = AsyncMock(
            return_value=FakeTestResult(
                failures=[FakeFailure("test_same")],
            )
        )

        analyzer = ImpactAnalyzer(runner)
        result = await analyzer.check_impact(["test_same"])

        assert result.has_regressions is False
        assert result.unchanged_failures == ["test_same"]
        assert result.resolved_failures == []
        assert result.new_failures == []

    @pytest.mark.asyncio
    async def test_passes_override_command(self):
        runner = MagicMock()
        runner.run_full = AsyncMock(
            return_value=FakeTestResult(failures=[])
        )

        analyzer = ImpactAnalyzer(runner)
        await analyzer.check_impact([], override_command="pytest tests/ -v")

        runner.run_full.assert_called_once_with(override_command="pytest tests/ -v")
