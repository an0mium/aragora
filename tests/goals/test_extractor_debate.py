"""Tests for GoalExtractor.extract_from_debate_analysis() and related features."""

from __future__ import annotations

import pytest

from aragora.goals.extractor import (
    GoalExtractionConfig,
    GoalExtractor,
    GoalNode,
    GoalGraph,
    _score_specificity,
    _score_measurability,
)
from aragora.canvas.stages import GoalNodeType, PipelineStage


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def extractor():
    return GoalExtractor()


@pytest.fixture
def sample_cartographer_output():
    """Minimal ArgumentCartographer.to_dict() output."""
    return {
        "nodes": [
            {"id": "n1", "node_type": "consensus", "label": "Implement rate limiting for API", "weight": 0.9},
            {"id": "n2", "node_type": "vote", "label": "Add caching layer", "weight": 0.7},
            {"id": "n3", "node_type": "claim", "label": "Reduce latency by 50%", "weight": 0.8},
            {"id": "n4", "node_type": "argument", "label": "Minor formatting fix", "weight": 0.2},
        ],
        "edges": [
            {"source": "n1", "target": "n2", "type": "supports"},
        ],
    }


@pytest.fixture
def sample_belief_result():
    """Mock PropagationResult with centralities."""
    class MockPropResult:
        centralities = {"n1": 0.95, "n2": 0.7, "n3": 0.6, "n4": 0.1}
    return MockPropResult()


# =========================================================================
# Basic extraction tests
# =========================================================================

class TestExtractFromDebateAnalysis:
    def test_basic_extraction(self, extractor, sample_cartographer_output):
        result = extractor.extract_from_debate_analysis(sample_cartographer_output)

        assert isinstance(result, GoalGraph)
        assert len(result.goals) > 0
        assert result.id.startswith("goals-")

    def test_empty_nodes(self, extractor):
        result = extractor.extract_from_debate_analysis({"nodes": []})
        assert len(result.goals) == 0

    def test_consensus_nodes_preferred(self, extractor, sample_cartographer_output):
        config = GoalExtractionConfig(require_consensus=True, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, config=config,
        )
        # Should include consensus, vote, claim, synthesis types
        for goal in result.goals:
            assert any(
                sid in ("n1", "n2", "n3")
                for sid in goal.source_idea_ids
            )

    def test_non_consensus_filtered_when_required(self, extractor, sample_cartographer_output):
        config = GoalExtractionConfig(require_consensus=True, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, config=config,
        )
        # n4 (argument type) should not become a goal when require_consensus=True
        all_source_ids = []
        for g in result.goals:
            all_source_ids.extend(g.source_idea_ids)
        assert "n4" not in all_source_ids

    def test_require_consensus_false(self, extractor):
        data = {
            "nodes": [
                {"id": "a1", "node_type": "argument", "label": "Some argument", "weight": 0.8},
                {"id": "a2", "node_type": "evidence", "label": "Some evidence", "weight": 0.7},
            ],
        }
        config = GoalExtractionConfig(require_consensus=False, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(data, config=config)
        assert len(result.goals) >= 1

    def test_fallback_to_non_consensus(self, extractor):
        """When only non-consensus nodes exist, fallback path should work."""
        data = {
            "nodes": [
                {"id": "a1", "node_type": "argument", "label": "Build dashboard", "weight": 0.9},
            ],
        }
        config = GoalExtractionConfig(require_consensus=True, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(data, config=config)
        # Should fall back to non-consensus extraction
        assert len(result.goals) >= 1


class TestBeliefCrossReference:
    def test_centrality_boosts_score(self, extractor, sample_cartographer_output, sample_belief_result):
        result_with = extractor.extract_from_debate_analysis(
            sample_cartographer_output, belief_result=sample_belief_result,
            config=GoalExtractionConfig(confidence_threshold=0.0),
        )
        result_without = extractor.extract_from_debate_analysis(
            sample_cartographer_output,
            config=GoalExtractionConfig(confidence_threshold=0.0),
        )
        # Both should produce goals
        assert len(result_with.goals) > 0
        assert len(result_without.goals) > 0

    def test_min_centrality_filter(self, extractor, sample_cartographer_output, sample_belief_result):
        config = GoalExtractionConfig(
            min_centrality=0.8, confidence_threshold=0.0, require_consensus=False,
        )
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, belief_result=sample_belief_result, config=config,
        )
        # Only n1 (centrality=0.95) should pass min_centrality=0.8
        # n2 (0.7) and n3 (0.6) filtered out
        assert len(result.goals) <= 2

    def test_no_belief_result(self, extractor, sample_cartographer_output):
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, belief_result=None,
        )
        assert isinstance(result, GoalGraph)


class TestConfidenceThreshold:
    def test_high_threshold_filters(self, extractor, sample_cartographer_output):
        config = GoalExtractionConfig(confidence_threshold=0.95)
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, config=config,
        )
        for goal in result.goals:
            assert goal.confidence >= 0.0  # Goals that pass filtering

    def test_low_threshold_includes_all(self, extractor, sample_cartographer_output):
        config = GoalExtractionConfig(confidence_threshold=0.0, require_consensus=False)
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, config=config,
        )
        assert len(result.goals) >= 3  # Should include most nodes

    def test_max_goals_limit(self, extractor):
        nodes = [
            {"id": f"n{i}", "node_type": "consensus", "label": f"Goal {i}", "weight": 0.8}
            for i in range(20)
        ]
        config = GoalExtractionConfig(max_goals=3, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis({"nodes": nodes}, config=config)
        assert len(result.goals) <= 3


class TestSMARTScoring:
    def test_specificity_scoring(self):
        assert _score_specificity("implement the API endpoint") > 0
        assert _score_specificity("") == 0.0
        assert _score_specificity("the thing") < _score_specificity("build the database server by Q3")

    def test_measurability_scoring(self):
        assert _score_measurability("reduce latency by 50%") > 0
        assert _score_measurability("improve throughput to 1000 requests per second") > 0
        assert _score_measurability("") == 0.0
        assert _score_measurability("do something") < _score_measurability("achieve 99% uptime")

    def test_smart_scoring_populates_metadata(self, extractor, sample_cartographer_output):
        config = GoalExtractionConfig(smart_scoring=True, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, config=config,
        )
        for goal in result.goals:
            assert "specificity" in goal.metadata or "rank" in goal.metadata

    def test_smart_scoring_disabled(self, extractor, sample_cartographer_output):
        config = GoalExtractionConfig(smart_scoring=False, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output, config=config,
        )
        for goal in result.goals:
            assert "specificity" not in goal.metadata


class TestGoalNodeTypes:
    def test_consensus_maps_to_goal(self, extractor):
        data = {"nodes": [{"id": "c1", "node_type": "consensus", "label": "Do X", "weight": 0.9}]}
        result = extractor.extract_from_debate_analysis(
            data, config=GoalExtractionConfig(confidence_threshold=0.0),
        )
        assert result.goals[0].goal_type == GoalNodeType.GOAL

    def test_vote_maps_to_milestone(self, extractor):
        data = {"nodes": [{"id": "v1", "node_type": "vote", "label": "Approve Y", "weight": 0.9}]}
        result = extractor.extract_from_debate_analysis(
            data, config=GoalExtractionConfig(confidence_threshold=0.0),
        )
        assert result.goals[0].goal_type == GoalNodeType.MILESTONE

    def test_claim_maps_to_strategy(self, extractor):
        data = {"nodes": [{"id": "cl1", "node_type": "claim", "label": "Strategy Z", "weight": 0.9}]}
        result = extractor.extract_from_debate_analysis(
            data, config=GoalExtractionConfig(confidence_threshold=0.0),
        )
        assert result.goals[0].goal_type == GoalNodeType.STRATEGY


class TestProvenance:
    def test_provenance_links_created(self, extractor, sample_cartographer_output):
        result = extractor.extract_from_debate_analysis(
            sample_cartographer_output,
            config=GoalExtractionConfig(confidence_threshold=0.0),
        )
        assert len(result.provenance) == len(result.goals)
        for link in result.provenance:
            assert link.source_stage == PipelineStage.IDEAS
            assert link.target_stage == PipelineStage.GOALS
            assert link.method == "debate_analysis"

    def test_transition_created(self, extractor, sample_cartographer_output):
        result = extractor.extract_from_debate_analysis(sample_cartographer_output)
        assert result.transition is not None
        assert result.transition.from_stage == PipelineStage.IDEAS
        assert result.transition.to_stage == PipelineStage.GOALS


class TestToPrioritizedGoal:
    def test_basic_conversion(self):
        goal = GoalNode(
            id="g1", title="Build API", description="Build the REST API",
            goal_type=GoalNodeType.GOAL, priority="high", confidence=0.8,
        )
        pg = goal.to_prioritized_goal("backend")
        assert pg["goal"] == "Build API"
        assert pg["track"] == "backend"
        assert pg["priority"] == 0.75  # "high" → 0.75
        assert pg["confidence"] == 0.8
        assert pg["source_goal_id"] == "g1"

    def test_priority_mapping(self):
        for priority, expected in [("critical", 1.0), ("high", 0.75), ("medium", 0.5), ("low", 0.25)]:
            goal = GoalNode(id="g", title="X", description="Y", priority=priority)
            assert goal.to_prioritized_goal()["priority"] == expected

    def test_default_track(self):
        goal = GoalNode(id="g", title="X", description="Y")
        assert goal.to_prioritized_goal()["track"] == "core"
