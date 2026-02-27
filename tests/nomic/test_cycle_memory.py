"""Tests for cross-cycle learning memory."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from aragora.nomic.cycle_memory import (
    CycleInsight,
    CycleOutcomeAnalyzer,
    FeedbackBridge,
    GoalTypeStats,
    StrategicMemoryStore,
    classify_goal_type,
)


class TestCycleInsight:
    """Tests for CycleInsight dataclass."""

    def test_defaults(self):
        insight = CycleInsight(
            cycle_id="cycle-1",
            goal_type="test_coverage",
            objective="Improve test coverage",
            outcome="succeeded",
            success_score=1.0,
        )
        assert insight.timestamp  # Auto-generated
        assert insight.files_changed == []
        assert insight.key_learnings == []

    def test_to_dict_roundtrip(self):
        insight = CycleInsight(
            cycle_id="cycle-1",
            goal_type="ux",
            objective="Improve onboarding",
            outcome="failed",
            success_score=0.0,
            failure_reason="Tests failed",
            key_learnings=["Need more test coverage"],
        )
        d = insight.to_dict()
        restored = CycleInsight.from_dict(d)
        assert restored.cycle_id == "cycle-1"
        assert restored.outcome == "failed"
        assert restored.failure_reason == "Tests failed"
        assert restored.key_learnings == ["Need more test coverage"]


class TestGoalTypeStats:
    """Tests for GoalTypeStats."""

    def test_success_rate_zero_attempts(self):
        stats = GoalTypeStats(goal_type="test")
        assert stats.success_rate == 0.0

    def test_success_rate(self):
        stats = GoalTypeStats(goal_type="test", total_attempts=10, successes=7, failures=3)
        assert stats.success_rate == 0.7

    def test_to_dict(self):
        stats = GoalTypeStats(
            goal_type="refactor",
            total_attempts=5,
            successes=3,
            failures=2,
            avg_score=0.65,
        )
        d = stats.to_dict()
        assert d["goal_type"] == "refactor"
        assert d["success_rate"] == 0.6


class TestClassifyGoalType:
    """Tests for goal type classification."""

    def test_test_coverage(self):
        assert classify_goal_type("Improve test coverage for debate module") == "test_coverage"

    def test_performance(self):
        assert classify_goal_type("Optimize query latency") == "performance"

    def test_ux(self):
        assert classify_goal_type("Improve user onboarding experience") == "ux"

    def test_refactor(self):
        assert classify_goal_type("Refactor the debate orchestrator") == "refactor"

    def test_security(self):
        assert classify_goal_type("Fix authentication vulnerability") == "security"

    def test_general_fallback(self):
        assert classify_goal_type("Do something completely unique") == "general"


class TestStrategicMemoryStore:
    """Tests for StrategicMemoryStore."""

    @pytest.fixture
    def store(self, tmp_path):
        return StrategicMemoryStore(store_path=tmp_path / "test_memory.json")

    def test_record_and_get_recent(self, store):
        insight = CycleInsight(
            cycle_id="cycle-1",
            goal_type="test_coverage",
            objective="Add tests",
            outcome="succeeded",
            success_score=1.0,
        )
        store.record(insight)

        recent = store.get_recent(limit=5)
        assert len(recent) == 1
        assert recent[0].cycle_id == "cycle-1"

    def test_persistence(self, tmp_path):
        path = tmp_path / "memory.json"

        # Write
        store1 = StrategicMemoryStore(store_path=path)
        store1.record(
            CycleInsight(
                cycle_id="cycle-1",
                goal_type="ux",
                objective="Fix dashboard",
                outcome="succeeded",
                success_score=0.9,
            )
        )

        # Read back
        store2 = StrategicMemoryStore(store_path=path)
        assert len(store2.get_recent()) == 1
        assert store2.get_recent()[0].cycle_id == "cycle-1"

    def test_find_similar(self, store):
        store.record(
            CycleInsight(
                cycle_id="c1",
                goal_type="test_coverage",
                objective="Improve test coverage for debate module",
                outcome="succeeded",
                success_score=1.0,
            )
        )
        store.record(
            CycleInsight(
                cycle_id="c2",
                goal_type="performance",
                objective="Optimize database query performance",
                outcome="failed",
                success_score=0.0,
            )
        )

        results = store.find_similar("Improve test coverage for pipeline")
        assert len(results) >= 1
        assert results[0].cycle_id == "c1"  # Most similar

    def test_find_similar_with_goal_type_filter(self, store):
        store.record(
            CycleInsight(
                cycle_id="c1",
                goal_type="test_coverage",
                objective="Improve test coverage",
                outcome="succeeded",
                success_score=1.0,
            )
        )
        store.record(
            CycleInsight(
                cycle_id="c2",
                goal_type="ux",
                objective="Improve user experience",
                outcome="succeeded",
                success_score=0.8,
            )
        )

        results = store.find_similar("Improve", goal_type="test_coverage")
        assert all(r.goal_type == "test_coverage" for r in results)

    def test_get_goal_type_stats(self, store):
        for i in range(5):
            store.record(
                CycleInsight(
                    cycle_id=f"c{i}",
                    goal_type="refactor",
                    objective=f"Refactor module {i}",
                    outcome="succeeded" if i < 3 else "failed",
                    success_score=1.0 if i < 3 else 0.0,
                    failure_reason="Test failures" if i >= 3 else "",
                    files_changed=["common_file.py"] if i >= 3 else [],
                )
            )

        stats = store.get_goal_type_stats("refactor")
        assert stats.total_attempts == 5
        assert stats.successes == 3
        assert stats.failures == 2
        assert stats.success_rate == 0.6
        assert "Test failures" in stats.common_failure_reasons

    def test_get_all_stats(self, store):
        store.record(
            CycleInsight(
                cycle_id="c1",
                goal_type="ux",
                objective="Fix dashboard",
                outcome="succeeded",
                success_score=1.0,
            )
        )
        store.record(
            CycleInsight(
                cycle_id="c2",
                goal_type="refactor",
                objective="Clean code",
                outcome="failed",
                success_score=0.0,
            )
        )

        all_stats = store.get_all_stats()
        assert "ux" in all_stats
        assert "refactor" in all_stats

    def test_risky_files_detection(self, store):
        # Same file appears in 2 failed cycles → risky
        for i in range(3):
            store.record(
                CycleInsight(
                    cycle_id=f"c{i}",
                    goal_type="refactor",
                    objective=f"Refactor {i}",
                    outcome="failed",
                    success_score=0.0,
                    files_changed=["dangerous_file.py", f"other_{i}.py"],
                )
            )

        stats = store.get_goal_type_stats("refactor")
        assert "dangerous_file.py" in stats.risky_files

    def test_corrupt_file_handled(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json{{{")

        store = StrategicMemoryStore(store_path=path)
        assert len(store.get_recent()) == 0  # Graceful recovery


class TestCycleOutcomeAnalyzer:
    """Tests for CycleOutcomeAnalyzer."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        store = StrategicMemoryStore(store_path=tmp_path / "test_memory.json")
        return CycleOutcomeAnalyzer(store=store)

    def test_analyze_successful_cycle(self, analyzer):
        insights = analyzer.analyze_cycle(
            cycle_id="cycle-1",
            goals=[{"title": "Add unit tests for debate module"}],
            results=[{"success": True, "tests_passed": 15, "tests_failed": 0}],
        )
        assert len(insights) == 1
        assert insights[0].outcome == "succeeded"
        assert insights[0].success_score == 1.0
        assert insights[0].goal_type == "test_coverage"

    def test_analyze_failed_cycle(self, analyzer):
        insights = analyzer.analyze_cycle(
            cycle_id="cycle-2",
            goals=[{"title": "Refactor orchestrator"}],
            results=[{"success": False, "error": "Import error in module"}],
        )
        assert len(insights) == 1
        assert insights[0].outcome == "failed"
        assert insights[0].success_score == 0.0
        assert "Import errors" in insights[0].key_learnings[0]

    def test_analyze_partial_success(self, analyzer):
        insights = analyzer.analyze_cycle(
            cycle_id="cycle-3",
            goals=[{"title": "Improve performance"}],
            results=[{"success": True, "tests_passed": 10, "tests_failed": 3}],
        )
        assert len(insights) == 1
        assert insights[0].outcome == "partial"
        assert 0.0 < insights[0].success_score < 1.0

    def test_analyze_with_metrics_delta(self, analyzer):
        insights = analyzer.analyze_cycle(
            cycle_id="cycle-4",
            goals=[{"title": "Add tests"}],
            results=[{"success": True, "tests_passed": 5, "tests_failed": 0}],
            metrics_before={"test_count": 100, "coverage": 0.6},
            metrics_after={"test_count": 115, "coverage": 0.65},
        )
        assert insights[0].metrics_delta["test_count"] == 15
        assert abs(insights[0].metrics_delta["coverage"] - 0.05) < 0.001

    def test_analyze_multiple_goals(self, analyzer):
        insights = analyzer.analyze_cycle(
            cycle_id="cycle-5",
            goals=[
                {"title": "Add tests"},
                {"title": "Fix security issue"},
            ],
            results=[
                {"success": True, "tests_passed": 5, "tests_failed": 0},
                {"success": False, "error": "Permission denied"},
            ],
        )
        assert len(insights) == 2
        assert insights[0].outcome == "succeeded"
        assert insights[1].outcome == "failed"

    def test_analyze_stores_insights(self, analyzer):
        analyzer.analyze_cycle(
            cycle_id="cycle-6",
            goals=[{"title": "Improve test coverage"}],
            results=[{"success": True, "tests_passed": 10, "tests_failed": 0}],
        )
        recent = analyzer.store.get_recent()
        assert len(recent) == 1

    def test_get_planning_context(self, analyzer):
        # Populate some history
        for i in range(5):
            analyzer.analyze_cycle(
                cycle_id=f"cycle-{i}",
                goals=[{"title": "Improve test coverage"}],
                results=[
                    {
                        "success": i < 2,  # 2 successes, 3 failures
                        "tests_passed": 5 if i < 2 else 0,
                        "tests_failed": 0 if i < 2 else 5,
                        "error": "Tests broke" if i >= 2 else "",
                    }
                ],
            )

        context = analyzer.get_planning_context("Improve test coverage")
        assert context["goal_type"] == "test_coverage"
        assert len(context["similar_past_cycles"]) > 0
        assert context["goal_type_stats"]["total_attempts"] == 5

    def test_get_planning_context_with_warnings(self, analyzer):
        # Create 3 failures for a goal type
        for i in range(4):
            analyzer.analyze_cycle(
                cycle_id=f"cycle-{i}",
                goals=[{"title": "Fix security auth issue"}],
                results=[{"success": False, "error": "Permission denied"}],
            )

        context = analyzer.get_planning_context("Fix security auth")
        assert any("success rate" in r.lower() for r in context["recommendations"])


class TestFeedbackBridge:
    """Tests for FeedbackBridge."""

    @pytest.fixture
    def bridge(self, tmp_path):
        store = StrategicMemoryStore(store_path=tmp_path / "test_memory.json")
        analyzer = CycleOutcomeAnalyzer(store=store)
        return FeedbackBridge(analyzer=analyzer)

    def test_generate_suggestions_from_failure(self, bridge):
        insights = [
            CycleInsight(
                cycle_id="cycle-1",
                goal_type="refactor",
                objective="Refactor orchestrator",
                outcome="failed",
                success_score=0.0,
                failure_reason="Tests broke",
                key_learnings=["Run tests first"],
            )
        ]
        suggestions = bridge.generate_improvement_suggestions(insights)
        assert len(suggestions) >= 1
        assert any("Retry" in s["task"] for s in suggestions)

    def test_generate_suggestions_from_partial(self, bridge):
        insights = [
            CycleInsight(
                cycle_id="cycle-2",
                goal_type="test_coverage",
                objective="Add tests",
                outcome="partial",
                success_score=0.7,
            )
        ]
        suggestions = bridge.generate_improvement_suggestions(insights)
        assert any("Complete" in s["task"] for s in suggestions)

    def test_meta_improvement_on_low_success_rate(self, bridge):
        # Record 3+ failures for a goal type first
        for i in range(4):
            bridge.analyzer.store.record(
                CycleInsight(
                    cycle_id=f"hist-{i}",
                    goal_type="performance",
                    objective=f"Optimize {i}",
                    outcome="failed",
                    success_score=0.0,
                )
            )

        insights = [
            CycleInsight(
                cycle_id="cycle-3",
                goal_type="performance",
                objective="Optimize queries",
                outcome="failed",
                success_score=0.0,
            )
        ]
        suggestions = bridge.generate_improvement_suggestions(insights)
        assert any("meta_improvement" == s["category"] for s in suggestions)

    def test_no_suggestions_for_success(self, bridge):
        insights = [
            CycleInsight(
                cycle_id="cycle-4",
                goal_type="docs",
                objective="Add documentation",
                outcome="succeeded",
                success_score=1.0,
            )
        ]
        suggestions = bridge.generate_improvement_suggestions(insights)
        # No retry/complete suggestions for successful cycles
        assert not any("Retry" in s.get("task", "") for s in suggestions)
