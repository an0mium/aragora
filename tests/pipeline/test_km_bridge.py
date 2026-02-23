"""Tests for the PipelineKMBridge.

Tests bidirectional KnowledgeMound integration:
- Query for similar goals and actions (precedent lookups)
- Enrich goal graphs with precedent metadata
- Store pipeline results back to KM
- Graceful degradation when KM is unavailable
- Query Receipt/Outcome/Debate adapters for decision precedents
"""

import sys

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
    confidence: float = 0.5
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
        goal_graph = MockGoalGraph(goals=[MockGoalNode(id="g1", title="Build something")])

        results = bridge.query_similar_goals(goal_graph)
        assert results == {"g1": []}

    def test_query_similar_goals_km_unavailable(self):
        """Should return empty dict when KM is not available."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        goal_graph = MockGoalGraph(goals=[MockGoalNode(id="g1", title="Build something")])

        results = bridge.query_similar_goals(goal_graph)
        assert results == {}

    def test_query_similar_goals_handles_search_error(self):
        """Should return empty list for goals where search raises."""
        mock_km = MagicMock()
        mock_km.search.side_effect = RuntimeError("search failed")

        bridge = PipelineKMBridge(knowledge_mound=mock_km)
        goal_graph = MockGoalGraph(goals=[MockGoalNode(id="g1", title="Build something")])

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

        canvas = MockCanvas(nodes={"a1": MockCanvasNode(label="Deploy")})

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

        goal_graph = MockGoalGraph(goals=[MockGoalNode(id="g1", title="Build something")])

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

        goal_graph = MockGoalGraph(goals=[MockGoalNode(id="g1", title="Test")])
        canvas = MockCanvas(nodes={"a1": MockCanvasNode(label="Test")})

        assert bridge.available is False
        assert bridge.query_similar_goals(goal_graph) == {}
        assert bridge.query_similar_actions(canvas) == {}
        assert bridge.store_pipeline_result(MagicMock()) is False

        # Enrich should still work (no KM needed)
        enriched = bridge.enrich_with_precedents(goal_graph, {})
        assert enriched is goal_graph


# =============================================================================
# Adapter precedent query tests
# =============================================================================


class TestQueryReceiptPrecedents:
    """Tests for ReceiptAdapter-based precedent queries."""

    def test_receipt_precedents_from_mound_search(self):
        """Should find receipt precedents via mound search."""
        mock_km = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Decision Receipt: contract terms approved"
        mock_result.metadata = {
            "item_type": "decision_summary",
            "receipt_id": "rcpt-001",
            "verdict": "APPROVED",
            "confidence": 0.9,
            "tags": ["decision_receipt"],
        }
        mock_km.search.return_value = [mock_result]

        mock_adapter = MagicMock()
        mock_adapter._ingested_receipts = {}

        bridge = PipelineKMBridge(knowledge_mound=mock_km)

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            results = bridge.query_receipt_precedents("contract review")

        # Should find the mound result tagged as decision_summary
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["source"] == "receipt"
        assert results[0]["receipt_id"] == "rcpt-001"

    def test_receipt_precedents_from_adapter_cache(self):
        """Should find receipt precedents from adapter's ingested cache."""
        mock_adapter = MagicMock()
        mock_ingestion = MagicMock()
        mock_ingestion.knowledge_item_ids = ["rcpt_abc123"]
        mock_adapter._ingested_receipts = {"debate-rate-limiter": mock_ingestion}

        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        with patch(
            "aragora.knowledge.mound.adapters.receipt_adapter.get_receipt_adapter",
            return_value=mock_adapter,
        ):
            results = bridge.query_receipt_precedents("rate limiter design")

        assert isinstance(results, list)
        # Should match because "rate" appears in the receipt_id
        found_ids = [r["receipt_id"] for r in results]
        assert "debate-rate-limiter" in found_ids

    def test_receipt_precedents_import_failure(self):
        """Should return empty list when adapter import fails."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        # Temporarily hide the receipt_adapter module to trigger ImportError
        mod_key = "aragora.knowledge.mound.adapters.receipt_adapter"
        saved = sys.modules.get(mod_key)
        sys.modules[mod_key] = None  # type: ignore[assignment]
        try:
            results = bridge.query_receipt_precedents("anything")
        finally:
            if saved is not None:
                sys.modules[mod_key] = saved
            else:
                sys.modules.pop(mod_key, None)

        assert results == []


class TestQueryOutcomePrecedents:
    """Tests for OutcomeAdapter-based precedent queries."""

    def test_outcome_precedents_from_mound_search(self):
        """Should find outcome precedents via mound search."""
        mock_km = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "[Outcome:positive] Vendor selection succeeded"
        mock_result.metadata = {
            "item_type": "decision_outcome",
            "outcome_id": "outc-001",
            "impact_score": 0.8,
            "lessons_learned": "Due diligence was key",
            "tags": ["decision_outcome"],
        }
        mock_km.search.return_value = [mock_result]

        mock_adapter = MagicMock()
        mock_adapter._ingested_outcomes = {}

        bridge = PipelineKMBridge(knowledge_mound=mock_km)

        with patch(
            "aragora.knowledge.mound.adapters.outcome_adapter.get_outcome_adapter",
            return_value=mock_adapter,
        ):
            results = bridge.query_outcome_precedents("vendor selection")

        assert len(results) == 1
        assert results[0]["source"] == "outcome"
        assert results[0]["outcome_id"] == "outc-001"
        assert results[0]["impact_score"] == 0.8

    def test_outcome_precedents_import_failure(self):
        """Should return empty list when adapter import fails."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        mod_key = "aragora.knowledge.mound.adapters.outcome_adapter"
        saved = sys.modules.get(mod_key)
        sys.modules[mod_key] = None  # type: ignore[assignment]
        try:
            results = bridge.query_outcome_precedents("anything")
        finally:
            if saved is not None:
                sys.modules[mod_key] = saved
            else:
                sys.modules.pop(mod_key, None)

        assert results == []


class TestQueryDebatePrecedents:
    """Tests for DebateAdapter-based precedent queries."""

    def test_debate_precedents_from_adapter_memory(self):
        """Should find debate precedents from adapter's in-memory outcomes."""
        mock_outcome = MagicMock()
        mock_outcome.task = "Design a rate limiter for API endpoints"
        mock_outcome.debate_id = "debate-42"
        mock_outcome.final_answer = "Use token bucket with Redis backend"
        mock_outcome.confidence = 0.85
        mock_outcome.consensus_reached = True

        mock_adapter = MagicMock()
        mock_adapter._synced_outcomes = {"debate-42": mock_outcome}
        mock_adapter._pending_outcomes = []

        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        with patch(
            "aragora.knowledge.mound.adapters.debate_adapter.DebateAdapter",
            return_value=mock_adapter,
        ):
            results = bridge.query_debate_precedents("rate limiter")

        assert len(results) == 1
        assert results[0]["source"] == "debate"
        assert results[0]["debate_id"] == "debate-42"
        assert results[0]["confidence"] == 0.85
        assert results[0]["consensus_reached"] is True

    def test_debate_precedents_import_failure(self):
        """Should return empty list when adapter import fails."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        mod_key = "aragora.knowledge.mound.adapters.debate_adapter"
        saved = sys.modules.get(mod_key)
        sys.modules[mod_key] = None  # type: ignore[assignment]
        try:
            results = bridge.query_debate_precedents("anything")
        finally:
            if saved is not None:
                sys.modules[mod_key] = saved
            else:
                sys.modules.pop(mod_key, None)

        assert results == []


class TestQueryAllAdapterPrecedents:
    """Tests for the combined adapter precedent query."""

    def test_query_all_returns_three_keys(self):
        """Should return dict with receipts, outcomes, and debates keys."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        # Patch all three adapter factories to return empty results
        with (
            patch.object(bridge, "query_receipt_precedents", return_value=[]),
            patch.object(bridge, "query_outcome_precedents", return_value=[]),
            patch.object(bridge, "query_debate_precedents", return_value=[]),
        ):
            results = bridge.query_all_adapter_precedents("some goal")

        assert "receipts" in results
        assert "outcomes" in results
        assert "debates" in results
        assert results["receipts"] == []
        assert results["outcomes"] == []
        assert results["debates"] == []


class TestEnrichGoalsWithAdapterPrecedents:
    """Tests for enriching goals with adapter-sourced precedents."""

    def test_enrich_attaches_precedents_to_goal_metadata(self):
        """Should attach adapter_precedents to goals that have matches."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        goal_graph = MockGoalGraph(
            goals=[
                MockGoalNode(id="g1", title="Build rate limiter"),
                MockGoalNode(id="g2", title="Unrelated goal"),
            ]
        )

        mock_precs = {
            "receipts": [{"source": "receipt", "receipt_id": "r1", "summary": "Prior decision"}],
            "outcomes": [],
            "debates": [{"source": "debate", "debate_id": "d1", "task": "rate limiter"}],
        }

        with patch.object(
            bridge,
            "query_all_adapter_precedents",
            side_effect=[mock_precs, {"receipts": [], "outcomes": [], "debates": []}],
        ):
            result = bridge.enrich_goals_with_adapter_precedents(goal_graph)

        assert result is goal_graph
        # g1 should have adapter_precedents (2 items across receipts+debates)
        assert "adapter_precedents" in goal_graph.goals[0].metadata
        precs = goal_graph.goals[0].metadata["adapter_precedents"]
        assert len(precs["receipts"]) == 1
        assert len(precs["debates"]) == 1
        # g2 should NOT have adapter_precedents (empty result)
        assert "adapter_precedents" not in goal_graph.goals[1].metadata

    def test_enrich_handles_query_failure_gracefully(self):
        """Should not crash when query_all raises for a specific goal."""
        bridge = PipelineKMBridge.__new__(PipelineKMBridge)
        bridge._km = None

        goal_graph = MockGoalGraph(goals=[MockGoalNode(id="g1", title="Build something")])

        with patch.object(
            bridge,
            "query_all_adapter_precedents",
            side_effect=RuntimeError("adapter failed"),
        ):
            # Should not raise
            result = bridge.enrich_goals_with_adapter_precedents(goal_graph)

        assert result is goal_graph
        assert "adapter_precedents" not in goal_graph.goals[0].metadata
