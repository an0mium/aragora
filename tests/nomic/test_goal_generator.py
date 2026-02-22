"""Tests for GoalGenerator — converts CodebaseHealthReport to PrioritizedGoals."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest


# Minimal stand-ins for assessment_engine types
def _make_candidate(
    description: str = "Fix something",
    priority: float = 0.8,
    source: str = "scanner",
    files: list[str] | None = None,
    category: str = "test",
):
    return types.SimpleNamespace(
        description=description,
        priority=priority,
        source=source,
        files=files or [],
        category=category,
    )


def _make_report(
    health_score: float = 0.7,
    candidates: list | None = None,
):
    return types.SimpleNamespace(
        health_score=health_score,
        improvement_candidates=candidates or [],
    )


class TestGoalGenerator:
    """Tests for GoalGenerator.generate_goals()."""

    def test_generate_goals_returns_prioritized_goals(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Fix 3 failing tests", 0.9, category="test"),
            _make_candidate("Reduce lint violations", 0.5, category="lint"),
        ])

        gen = GoalGenerator()
        goals = gen.generate_goals(report)

        assert len(goals) == 2
        assert goals[0].priority == 1
        assert goals[1].priority == 2

    def test_generate_goals_empty_candidates(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[])
        gen = GoalGenerator()
        goals = gen.generate_goals(report)

        assert goals == []

    def test_generate_goals_respects_max_goals(self):
        from aragora.nomic.goal_generator import GoalGenerator

        candidates = [_make_candidate(f"Issue {i}", 0.8 - i * 0.1) for i in range(10)]
        report = _make_report(candidates=candidates)

        gen = GoalGenerator(max_goals=3)
        goals = gen.generate_goals(report)

        assert len(goals) == 3

    def test_generate_goals_maps_category_to_track(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Test issue", category="test"),
            _make_candidate("Lint issue", category="lint"),
            _make_candidate("TODO cleanup", category="todo"),
        ])

        gen = GoalGenerator()
        goals = gen.generate_goals(report)

        # test → qa, lint → core, todo → developer
        assert goals[0].track.value == "qa"
        assert goals[1].track.value == "core"
        assert goals[2].track.value == "developer"

    def test_generate_goals_maps_category_to_impact(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Regression", category="regression"),
            _make_candidate("TODO", category="todo"),
        ])

        gen = GoalGenerator()
        goals = gen.generate_goals(report)

        assert goals[0].estimated_impact == "high"
        assert goals[1].estimated_impact == "low"

    def test_generate_goals_includes_file_hints(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Fix tests", files=["tests/test_foo.py", "tests/test_bar.py"]),
        ])

        gen = GoalGenerator()
        goals = gen.generate_goals(report)

        assert goals[0].file_hints == ["tests/test_foo.py", "tests/test_bar.py"]

    def test_generate_goals_graceful_on_import_error(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[_make_candidate()])
        gen = GoalGenerator()

        with patch.dict("sys.modules", {"aragora.nomic.meta_planner": None}):
            goals = gen.generate_goals(report)
            assert goals == []

    def test_generate_goals_unknown_category_defaults_to_core(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Unknown thing", category="unknown_cat"),
        ])

        gen = GoalGenerator()
        goals = gen.generate_goals(report)

        assert goals[0].track.value == "core"


class TestGoalGeneratorIdeas:
    """Tests for GoalGenerator.generate_ideas()."""

    def test_generate_ideas_returns_strings(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Fix tests", category="test"),
        ])

        gen = GoalGenerator()
        ideas = gen.generate_ideas(report)

        assert len(ideas) == 1
        assert "[test]" in ideas[0]
        assert "Fix tests" in ideas[0]

    def test_generate_ideas_includes_files(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Fix foo", files=["foo.py"]),
        ])

        gen = GoalGenerator()
        ideas = gen.generate_ideas(report)

        assert "foo.py" in ideas[0]

    def test_generate_ideas_empty_candidates(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[])
        gen = GoalGenerator()

        assert gen.generate_ideas(report) == []


class TestGoalGeneratorObjective:
    """Tests for GoalGenerator.generate_objective()."""

    def test_generate_objective_from_top_candidate(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[
            _make_candidate("Fix 5 failing tests", 0.9),
        ])

        gen = GoalGenerator()
        obj = gen.generate_objective(report)

        assert "[auto-assess]" in obj
        assert "Fix 5 failing tests" in obj

    def test_generate_objective_no_candidates(self):
        from aragora.nomic.goal_generator import GoalGenerator

        report = _make_report(candidates=[])
        gen = GoalGenerator()
        obj = gen.generate_objective(report)

        assert "no issues" in obj.lower()
