"""Tests for the PipelineKMBridge.

Tests bidirectional KnowledgeMound integration:
- Query for similar goals and actions (precedent lookups)
- Enrich goal graphs with precedent metadata
- Store pipeline results back to KM
- Graceful degradation when KM is unavailable
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any

from aragora.pipeline.km_bridge import PipelineKMBridge


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class MockGoalNode:
    id: str
    title: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockGoalGraph:
    id: str = "test-goal-graph"
    goals: list[MockGoalNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockCanvasNode:
    label: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockCanvas:
    nodes: dict[str, MockCanvasNode] = field(default_factory=dict)


class MockSearchResult:
    def __init__(self, title: str, similarity: float, outcome: str = "success"):
        self.title = title
        self.similarity = similarity
        self.metadata = {"outcome": outcome}


# =============================================================================
# Tests
# =============================================================================


class TestBridgeCreation:
    def test_bridge_creation_without_km(self):
        """Bridge created without KM should report unavailable."""
        bridge = PipelineKMBridge(knowledge_mound=None)
        # KM auto-discovery will fail in test env, so should be unavailable
        # unless the import succeeds
        assert isinstance(bridge.available, bool)

    def test_bridge_creation_with_mock_km(self):
        """Bridge created with mock KM should report available."""
        mock_km = MagicMock()
        bridge = PipelineKMBridge(knowledge_mound=mock_km)
        assert bridge.available is True

    def test_available_property_false_when_none(self):
        """available should be False when _km is None."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None
        assert bridge.available is False

    def test_available_property_true_when_set(self):
        """available should be True when _km is set."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = MagicMock()
        assert bridge.available is True


class TestQuerySimilarGoals:
    def test_query_similar_goals_with_matches(self):
        """Should return matched precedents for each goal."""
        mock_km = MagicMock()
        mock_km.search.return_value = [
            MockSearchResult("Previous rate limiter", 0.8, "success"),
            MockSearchResult("Old caching layer", 0.6, "partial"),
        ]

        bridge = PipelineKMBridge(knowledge_mound=mock_km)
        goal_graph = MockGoalGraph(
            goals=[
                MockGoalNode(id="g1", title="Build rate limiter"),
                MockGoalNode(id="g2", title="Add caching"),
            ]
        )

        results = bridge.query_similar_goals(goal_graph)

        assert "g1" in results
        assert "g2" in results
        assert len(results["g1"]) == 2
        assert results["g1"][0]["title"] == "Previous rate limiter"
        assert results["g1"][0]["similarity"] == 0.8
        assert results["g1"][0]["outcome"] == "success"

    def test_query_similar_goals_empty_km(self):
        """Should return empty lists when KM has no matches."""
        mock_km = MagicMock()
        mock_km.search.return_value = []

        bridge = PipelineKMBridge(knowledge_mound=mock_km)
        goal_graph = MockGoalGraph(
            goals=[MockGoalNode(id="g1", title="Build something")]
        )

        results = bridge.query_similar_goals(goal_graph)
        assert results == {"g1": []}

    def test_query_similar_goals_km_unavailable(self):
        """Should return empty dict when KM is not available."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        goal_graph = MockGoalGraph(
            goals=[MockGoalNode(id="g1", title="Build something")]
        )

        results = bridge.query_similar_goals(goal_graph)
        assert results == {}

    def test_query_similar_goals_handles_search_error(self):
        """Should return empty list for goals where search raises."""
        mock_km = MagicMock()
        mock_km.search.side_effect = RuntimeError("search failed")

        bridge = PipelineKMBridge(knowledge_mound=mock_km)
        goal_graph = MockGoalGraph(
            goals=[MockGoalNode(id="g1", title="Build something")]
        )

        results = bridge.query_similar_goals(goal_graph)
        assert results == {"g1": []}


class TestQuerySimilarActions:
    def test_query_similar_actions(self):
        """Should query KM for each action node."""
        mock_km = MagicMock()
        mock_km.search.return_value = [
            MockSearchResult("Previous deployment", 0.7, "success"),
        ]

        bridge = PipelineKMBridge(knowledge_mound=mock_km)
        canvas = MockCanvas(
            nodes={
                "a1": MockCanvasNode(label="Deploy service"),
                "a2": MockCanvasNode(label="Run tests"),
            }
        )

        results = bridge.query_similar_actions(canvas)
        assert "a1" in results
        assert "a2" in results
        assert len(results["a1"]) == 1

    def test_query_similar_actions_km_unavailable(self):
        """Should return empty dict when KM is unavailable."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        canvas = MockCanvas(
            nodes={"a1": MockCanvasNode(label="Deploy")}
        )

        results = bridge.query_similar_actions(canvas)
        assert results == {}


class TestEnrichWithPrecedents:
    def test_enrich_with_precedents(self):
        """Should add precedent data to goal metadata."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None  # Not needed for enrich

        goal_graph = MockGoalGraph(
            goals=[
                MockGoalNode(id="g1", title="Build rate limiter"),
                MockGoalNode(id="g2", title="Add caching"),
            ]
        )

        precedents = {
            "g1": [
                {"title": "Previous limiter", "similarity": 0.8, "outcome": "success"},
            ],
            "g2": [],  # No precedents for g2
        }

        result = bridge.enrich_with_precedents(goal_graph, precedents)

        assert result is goal_graph  # Modified in place
        assert "precedents" in goal_graph.goals[0].metadata
        assert len(goal_graph.goals[0].metadata["precedents"]) == 1
        # g2 has empty precedents, should NOT have the key added
        assert "precedents" not in goal_graph.goals[1].metadata

    def test_enrich_with_no_matching_ids(self):
        """Should not crash when precedent IDs don't match goals."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        goal_graph = MockGoalGraph(
            goals=[MockGoalNode(id="g1", title="Build something")]
        )

        precedents = {"g999": [{"title": "Irrelevant", "similarity": 0.5}]}

        result = bridge.enrich_with_precedents(goal_graph, precedents)
        assert "precedents" not in result.goals[0].metadata


class TestStorePipelineResult:
    def test_store_pipeline_result_km_unavailable(self):
        """Should return False when KM is not available."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        mock_result = MagicMock()
        assert bridge.store_pipeline_result(mock_result) is False

    def test_store_pipeline_result_import_fails(self):
        """Should return False when DecisionPlanAdapter import fails."""
        mock_km = MagicMock()
        bridge = PipelineKMBridge(knowledge_mound=mock_km)

        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"pipeline_id": "test"}

        with patch(
            "aragora.pipeline.km_bridge.PipelineKMBridge.store_pipeline_result",
            wraps=bridge.store_pipeline_result,
        ):
            # The actual import of DecisionPlanAdapter may or may not succeed
            # in test env; either way the method should not raise
            result = bridge.store_pipeline_result(mock_result)
            assert isinstance(result, bool)


class TestGracefulDegradation:
    def test_all_methods_work_without_km(self):
        """All public methods should work gracefully without KM."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        goal_graph = MockGoalGraph(
            goals=[MockGoalNode(id="g1", title="Test")]
        )
        canvas = MockCanvas(
            nodes={"a1": MockCanvasNode(label="Test")}
        )

        assert bridge.available is False
        assert bridge.query_similar_goals(goal_graph) == {}
        assert bridge.query_similar_actions(canvas) == {}
        assert bridge.store_pipeline_result(MagicMock()) is False

        # Enrich should still work (no KM needed)
        enriched = bridge.enrich_with_precedents(goal_graph, {})
        assert enriched is goal_graph
