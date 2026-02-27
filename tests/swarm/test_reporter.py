"""Tests for SwarmReporter and SwarmReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.swarm.reporter import SwarmReport, SwarmReporter
from aragora.swarm.spec import SwarmSpec


@dataclass
class MockAssignment:
    """Mock assignment for testing."""

    subtask_title: str = "Test task"
    status: str = "completed"
    error: str = ""


@dataclass
class MockResult:
    """Mock OrchestrationResult for testing."""

    total_subtasks: int = 5
    completed_subtasks: int = 4
    failed_subtasks: int = 1
    skipped_subtasks: int = 0
    assignments: list[Any] = field(default_factory=list)
    total_cost_usd: float = 2.50


class TestSwarmReport:
    """Test SwarmReport rendering."""

    def test_plain_text_success(self):
        report = SwarmReport(
            success=True,
            summary="Everything worked great.",
            what_was_done=["Fixed the login", "Updated the tests"],
            what_failed=[],
            what_to_do_next=["Review changes"],
            duration_seconds=120.0,
            budget_spent_usd=1.50,
        )
        text = report.to_plain_text()
        assert "SUCCESS" in text
        assert "Everything worked great." in text
        assert "Fixed the login" in text
        assert "Updated the tests" in text
        assert "$1.50" in text
        assert "2m 0s" in text

    def test_plain_text_failure(self):
        report = SwarmReport(
            success=False,
            summary="Some tasks failed.",
            what_was_done=["Task A"],
            what_failed=["Task B: timeout"],
            what_to_do_next=["Retry task B"],
        )
        text = report.to_plain_text()
        assert "ISSUES" in text
        assert "Task B: timeout" in text

    def test_markdown_rendering(self):
        report = SwarmReport(
            success=True,
            summary="All good.",
            what_was_done=["Item 1"],
            what_to_do_next=["Review"],
        )
        md = report.to_markdown()
        assert "# Swarm Report" in md
        assert "- Item 1" in md

    def test_to_dict(self):
        spec = SwarmSpec(raw_goal="test")
        report = SwarmReport(
            success=True,
            summary="Test",
            spec=spec,
            duration_seconds=60.0,
        )
        data = report.to_dict()
        assert data["success"] is True
        assert data["summary"] == "Test"
        assert data["spec"]["raw_goal"] == "test"

    def test_to_dict_without_spec(self):
        report = SwarmReport(success=False, summary="No spec")
        data = report.to_dict()
        assert data["spec"] is None


class TestSwarmReporter:
    """Test SwarmReporter template-based generation."""

    @pytest.mark.asyncio
    async def test_template_report_success(self):
        spec = SwarmSpec(refined_goal="Fix all bugs")
        result = MockResult(
            total_subtasks=3,
            completed_subtasks=3,
            failed_subtasks=0,
            assignments=[
                MockAssignment(subtask_title="Fix bug A", status="completed"),
                MockAssignment(subtask_title="Fix bug B", status="completed"),
                MockAssignment(subtask_title="Fix bug C", status="completed"),
            ],
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result, duration_seconds=90.0)

        assert report.success is True
        assert "great news" in report.summary.lower()
        assert "3" in report.summary
        assert len(report.what_was_done) == 3

    @pytest.mark.asyncio
    async def test_template_report_partial_failure(self):
        spec = SwarmSpec(refined_goal="Improve everything")
        result = MockResult(
            total_subtasks=5,
            completed_subtasks=3,
            failed_subtasks=2,
            assignments=[
                MockAssignment(subtask_title="Task A", status="completed"),
                MockAssignment(subtask_title="Task B", status="failed", error="timeout"),
            ],
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is False
        assert "3" in report.summary and "5" in report.summary

    @pytest.mark.asyncio
    async def test_template_report_total_failure(self):
        spec = SwarmSpec(refined_goal="Do stuff")
        result = MockResult(
            total_subtasks=2,
            completed_subtasks=0,
            failed_subtasks=2,
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is False
        assert "wasn't able to complete" in report.summary

    @pytest.mark.asyncio
    async def test_template_report_with_skipped(self):
        spec = SwarmSpec(refined_goal="Partial work")
        result = MockResult(
            total_subtasks=4,
            completed_subtasks=2,
            failed_subtasks=0,
            skipped_subtasks=2,
        )

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is True
        assert any("skipped" in item.lower() for item in report.what_to_do_next)

    @pytest.mark.asyncio
    async def test_budget_extraction(self):
        spec = SwarmSpec(raw_goal="Budget test")
        result = MockResult(total_cost_usd=3.75)

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.budget_spent_usd == 3.75

    @pytest.mark.asyncio
    async def test_empty_result(self):
        spec = SwarmSpec(raw_goal="Nothing happened")
        result = MockResult(total_subtasks=0, completed_subtasks=0, failed_subtasks=0)

        reporter = SwarmReporter()
        report = await reporter.generate(spec, result)

        assert report.success is False
