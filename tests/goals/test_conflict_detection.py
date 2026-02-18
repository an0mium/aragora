"""Tests for goal conflict detection and enhanced SMART scoring.

Covers: contradictory keywords, circular dependencies, near-duplicate detection,
SMART dimension scoring, and improvement suggestions.
"""

from __future__ import annotations

import pytest

from aragora.canvas.stages import GoalNodeType
from aragora.goals.extractor import (
    GoalExtractor,
    GoalGraph,
    GoalNode,
    _score_achievability,
    _score_relevance,
    _score_time_bound,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor():
    return GoalExtractor()


def _make_goal(
    id: str = "g1",
    title: str = "Test goal",
    description: str = "",
    dependencies: list[str] | None = None,
    source_idea_ids: list[str] | None = None,
    goal_type: GoalNodeType = GoalNodeType.GOAL,
) -> GoalNode:
    return GoalNode(
        id=id,
        title=title,
        description=description,
        dependencies=dependencies or [],
        source_idea_ids=source_idea_ids or [],
        goal_type=goal_type,
    )


def _make_graph(*goals: GoalNode) -> GoalGraph:
    return GoalGraph(id="test-graph", goals=list(goals))


# ===========================================================================
# TestContradictoryGoals
# ===========================================================================


class TestContradictoryGoals:
    def test_maximize_vs_minimize(self, extractor):
        g1 = _make_goal("g1", "Maximize throughput", "maximize the system throughput")
        g2 = _make_goal("g2", "Minimize throughput", "minimize the system throughput")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert len(contradictions) >= 1
        assert contradictions[0]["severity"] == "high"

    def test_increase_vs_decrease(self, extractor):
        g1 = _make_goal("g1", "Increase test coverage")
        g2 = _make_goal("g2", "Decrease test coverage")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert len(contradictions) >= 1

    def test_add_vs_remove(self, extractor):
        g1 = _make_goal("g1", "Add logging module")
        g2 = _make_goal("g2", "Remove logging module")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert len(contradictions) >= 1

    def test_enable_vs_disable(self, extractor):
        g1 = _make_goal("g1", "Enable caching layer")
        g2 = _make_goal("g2", "Disable caching layer")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert len(contradictions) >= 1

    def test_no_false_positives_on_unrelated(self, extractor):
        g1 = _make_goal("g1", "Improve database performance")
        g2 = _make_goal("g2", "Design frontend components")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert len(contradictions) == 0

    def test_contradiction_in_description(self, extractor):
        g1 = _make_goal("g1", "Update system", "increase the memory limit")
        g2 = _make_goal("g2", "Tune system", "decrease the memory limit")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert len(contradictions) >= 1

    def test_conflict_has_both_goal_ids(self, extractor):
        g1 = _make_goal("goal-a", "Maximize efficiency")
        g2 = _make_goal("goal-b", "Minimize efficiency")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert len(contradictions) >= 1
        assert "goal-a" in contradictions[0]["goal_ids"]
        assert "goal-b" in contradictions[0]["goal_ids"]


# ===========================================================================
# TestCircularDependencies
# ===========================================================================


class TestCircularDependencies:
    def test_direct_cycle(self, extractor):
        g1 = _make_goal("g1", "Goal A", dependencies=["g2"])
        g2 = _make_goal("g2", "Goal B", dependencies=["g1"])
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        cycles = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(cycles) >= 1
        assert cycles[0]["severity"] == "high"

    def test_indirect_cycle(self, extractor):
        g1 = _make_goal("g1", "Goal A", dependencies=["g2"])
        g2 = _make_goal("g2", "Goal B", dependencies=["g3"])
        g3 = _make_goal("g3", "Goal C", dependencies=["g1"])
        graph = _make_graph(g1, g2, g3)
        conflicts = extractor.detect_goal_conflicts(graph)
        cycles = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(cycles) >= 1
        # All three should be in the cycle
        cycle_ids = set(cycles[0]["goal_ids"])
        assert {"g1", "g2", "g3"}.issubset(cycle_ids)

    def test_no_cycle(self, extractor):
        g1 = _make_goal("g1", "Goal A", dependencies=["g2"])
        g2 = _make_goal("g2", "Goal B", dependencies=["g3"])
        g3 = _make_goal("g3", "Goal C")
        graph = _make_graph(g1, g2, g3)
        conflicts = extractor.detect_goal_conflicts(graph)
        cycles = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(cycles) == 0

    def test_self_dependency(self, extractor):
        g1 = _make_goal("g1", "Goal A", dependencies=["g1"])
        graph = _make_graph(g1)
        conflicts = extractor.detect_goal_conflicts(graph)
        cycles = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(cycles) >= 1

    def test_cycle_with_description(self, extractor):
        g1 = _make_goal("g1", "First goal", dependencies=["g2"])
        g2 = _make_goal("g2", "Second goal", dependencies=["g1"])
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        cycles = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(cycles) >= 1
        assert "suggestion" in cycles[0]
        assert len(cycles[0]["description"]) > 0

    def test_external_dependency_ignored(self, extractor):
        """Dependencies referencing goals not in the graph should be ignored."""
        g1 = _make_goal("g1", "Goal A", dependencies=["nonexistent"])
        graph = _make_graph(g1)
        conflicts = extractor.detect_goal_conflicts(graph)
        cycles = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert len(cycles) == 0


# ===========================================================================
# TestNearDuplicates
# ===========================================================================


class TestNearDuplicates:
    def test_similar_titles_detected(self, extractor):
        g1 = _make_goal("g1", "Optimize database query performance")
        g2 = _make_goal("g2", "Optimize database query speed performance")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        dupes = [c for c in conflicts if c["type"] == "near_duplicate"]
        assert len(dupes) >= 1
        assert dupes[0]["severity"] == "medium"

    def test_different_titles_not_flagged(self, extractor):
        g1 = _make_goal("g1", "Optimize database queries")
        g2 = _make_goal("g2", "Design frontend components")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        dupes = [c for c in conflicts if c["type"] == "near_duplicate"]
        assert len(dupes) == 0

    def test_identical_titles_detected(self, extractor):
        g1 = _make_goal("g1", "Improve server performance")
        g2 = _make_goal("g2", "Improve server performance")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        dupes = [c for c in conflicts if c["type"] == "near_duplicate"]
        assert len(dupes) >= 1


# ===========================================================================
# TestConflictSeverity
# ===========================================================================


class TestConflictSeverity:
    def test_contradiction_is_high(self, extractor):
        g1 = _make_goal("g1", "Maximize throughput")
        g2 = _make_goal("g2", "Minimize throughput")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        contradictions = [c for c in conflicts if c["type"] == "contradiction"]
        assert all(c["severity"] == "high" for c in contradictions)

    def test_circular_dependency_is_high(self, extractor):
        g1 = _make_goal("g1", "A", dependencies=["g2"])
        g2 = _make_goal("g2", "B", dependencies=["g1"])
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        cycles = [c for c in conflicts if c["type"] == "circular_dependency"]
        assert all(c["severity"] == "high" for c in cycles)

    def test_near_duplicate_is_medium(self, extractor):
        g1 = _make_goal("g1", "Optimize database query performance")
        g2 = _make_goal("g2", "Optimize database query speed performance")
        graph = _make_graph(g1, g2)
        conflicts = extractor.detect_goal_conflicts(graph)
        dupes = [c for c in conflicts if c["type"] == "near_duplicate"]
        assert all(c["severity"] == "medium" for c in dupes)

    def test_empty_graph_no_conflicts(self, extractor):
        graph = _make_graph()
        conflicts = extractor.detect_goal_conflicts(graph)
        assert conflicts == []


# ===========================================================================
# TestSMARTScoring
# ===========================================================================


class TestSMARTScoring:
    def test_all_dimensions_present(self, extractor):
        goal = _make_goal("g1", "Implement API endpoint", "Add REST endpoint")
        scores = extractor.score_smart(goal)
        assert "specific" in scores
        assert "measurable" in scores
        assert "achievable" in scores
        assert "relevant" in scores
        assert "time_bound" in scores
        assert "overall" in scores

    def test_scores_between_0_and_1(self, extractor):
        goal = _make_goal("g1", "Implement API endpoint by Q2 2026")
        scores = extractor.score_smart(goal)
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{key} score out of range: {value}"

    def test_specific_goal_scores_high_specificity(self, extractor):
        goal = _make_goal(
            "g1",
            "Implement database migration endpoint by Friday",
            "Build a REST API endpoint for database schema migration",
        )
        scores = extractor.score_smart(goal)
        assert scores["specific"] > 0.3

    def test_measurable_goal_scores_high(self, extractor):
        goal = _make_goal(
            "g1",
            "Reduce API latency by 50%",
            "Decrease average response latency to under 100ms",
        )
        scores = extractor.score_smart(goal)
        assert scores["measurable"] > 0.3

    def test_achievable_scoped_goal(self, extractor):
        goal = _make_goal(
            "g1",
            "Implement single authentication module",
            "Add one specific component for auth",
        )
        scores = extractor.score_smart(goal)
        assert scores["achievable"] > 0.3

    def test_overambitious_goal_low_achievability(self, extractor):
        goal = _make_goal(
            "g1",
            "Transform everything about the entire system",
            "Revolutionize all components universally",
        )
        scores = extractor.score_smart(goal)
        assert scores["achievable"] < 0.3

    def test_time_bound_goal(self, extractor):
        goal = _make_goal(
            "g1",
            "Deploy feature within 2 sprints by Q2",
            "Complete deployment before the deadline",
        )
        scores = extractor.score_smart(goal)
        assert scores["time_bound"] > 0.3

    def test_no_time_reference_low_score(self, extractor):
        goal = _make_goal("g1", "Improve code quality")
        scores = extractor.score_smart(goal)
        assert scores["time_bound"] < 0.3

    def test_overall_weighted_average(self, extractor):
        goal = _make_goal("g1", "Test goal")
        scores = extractor.score_smart(goal)
        expected = (
            0.25 * scores["specific"]
            + 0.25 * scores["measurable"]
            + 0.20 * scores["achievable"]
            + 0.15 * scores["relevant"]
            + 0.15 * scores["time_bound"]
        )
        assert scores["overall"] == pytest.approx(expected)

    def test_source_ideas_boost_relevance(self, extractor):
        goal_with = _make_goal("g1", "Improve API", source_idea_ids=["idea-1", "idea-2"])
        goal_without = _make_goal("g2", "Improve API")
        scores_with = extractor.score_smart(goal_with)
        scores_without = extractor.score_smart(goal_without)
        assert scores_with["relevant"] >= scores_without["relevant"]

    def test_empty_goal(self, extractor):
        goal = _make_goal("g1", "", "")
        scores = extractor.score_smart(goal)
        # All scores should be 0 for empty text (except maybe tiny float)
        assert scores["overall"] == pytest.approx(0.0, abs=0.01)


# ===========================================================================
# TestSuggestImprovements
# ===========================================================================


class TestSuggestImprovements:
    def test_vague_goal_gets_suggestions(self, extractor):
        goal = _make_goal("g1", "Make things better")
        suggestions = extractor.suggest_improvements(goal)
        assert len(suggestions) >= 1

    def test_specific_suggestion_for_low_specificity(self, extractor):
        goal = _make_goal("g1", "Make things better")
        suggestions = extractor.suggest_improvements(goal)
        assert any("specific" in s.lower() or "technical" in s.lower() for s in suggestions)

    def test_measurable_suggestion(self, extractor):
        goal = _make_goal("g1", "Make things better")
        suggestions = extractor.suggest_improvements(goal)
        assert any("quantitative" in s.lower() or "criteria" in s.lower() for s in suggestions)

    def test_time_suggestion(self, extractor):
        goal = _make_goal("g1", "Build an API endpoint module")
        suggestions = extractor.suggest_improvements(goal)
        assert any("timeline" in s.lower() or "deadline" in s.lower() for s in suggestions)

    def test_well_formed_goal_fewer_suggestions(self, extractor):
        goal = _make_goal(
            "g1",
            "Implement single API endpoint to reduce latency by 50% within sprint",
            "Build one specific module for the database service before the Q2 deadline",
            source_idea_ids=["idea-1"],
        )
        suggestions = extractor.suggest_improvements(goal)
        # Well-formed goal should have fewer suggestions than a vague one
        vague_goal = _make_goal("g2", "Make things better")
        vague_suggestions = extractor.suggest_improvements(vague_goal)
        assert len(suggestions) <= len(vague_suggestions)

    def test_returns_list_of_strings(self, extractor):
        goal = _make_goal("g1", "Some goal")
        suggestions = extractor.suggest_improvements(goal)
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert isinstance(s, str)


# ===========================================================================
# TestScoringHelpers
# ===========================================================================


class TestScoringHelpers:
    def test_achievability_empty(self):
        assert _score_achievability("") == 0.0

    def test_achievability_scoped(self):
        score = _score_achievability("implement a single module for this specific task")
        assert score > 0.0

    def test_achievability_overambitious(self):
        score = _score_achievability("revolutionize everything in the entire universal system")
        # Over-ambition penalty should make this low
        assert score < 0.3

    def test_time_bound_empty(self):
        assert _score_time_bound("") == 0.0

    def test_time_bound_with_deadline(self):
        score = _score_time_bound("complete by Q2 within the sprint timeline")
        assert score > 0.3

    def test_relevance_empty(self):
        assert _score_relevance("") == 0.0

    def test_relevance_technical(self):
        score = _score_relevance("deploy API service to production server")
        assert score > 0.0

    def test_relevance_boost_from_sources(self):
        score_with = _score_relevance("improve API", ["idea-1"])
        score_without = _score_relevance("improve API")
        assert score_with >= score_without


# ===========================================================================
# TestConflictIntegration
# ===========================================================================


class TestConflictIntegration:
    def test_mixed_conflicts(self, extractor):
        """A graph with multiple conflict types."""
        g1 = _make_goal("g1", "Maximize system throughput", dependencies=["g2"])
        g2 = _make_goal("g2", "Minimize system throughput", dependencies=["g1"])
        g3 = _make_goal("g3", "Optimize database query performance")
        g4 = _make_goal("g4", "Optimize database query speed performance")
        graph = _make_graph(g1, g2, g3, g4)

        conflicts = extractor.detect_goal_conflicts(graph)

        types = {c["type"] for c in conflicts}
        # Should detect contradiction (maximize vs minimize)
        assert "contradiction" in types
        # Should detect circular dependency (g1 <-> g2)
        assert "circular_dependency" in types
        # Should detect near-duplicate (g3 vs g4)
        assert "near_duplicate" in types

    def test_all_conflicts_have_required_keys(self, extractor):
        g1 = _make_goal("g1", "Increase speed", dependencies=["g2"])
        g2 = _make_goal("g2", "Decrease speed", dependencies=["g1"])
        graph = _make_graph(g1, g2)

        conflicts = extractor.detect_goal_conflicts(graph)
        for c in conflicts:
            assert "type" in c
            assert "severity" in c
            assert "goal_ids" in c
            assert "description" in c
            assert "suggestion" in c
            assert isinstance(c["goal_ids"], list)
            assert len(c["goal_ids"]) >= 1
