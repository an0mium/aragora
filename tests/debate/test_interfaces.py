"""
Tests for debate protocol interfaces.

Tests cover:
- PositionTrackerProtocol
- CalibrationTrackerProtocol
- BeliefNetworkProtocol
- BeliefPropagationAnalyzerProtocol
- CitationExtractorProtocol
- InsightExtractorProtocol
- InsightStoreProtocol
- CritiqueStoreProtocol
- ArgumentCartographerProtocol

Validates that:
- Protocols are runtime-checkable
- Minimal implementations satisfy protocol requirements
- Protocol methods have correct signatures
"""

from __future__ import annotations

from typing import Any

import pytest

from aragora.debate.interfaces import (
    ArgumentCartographerProtocol,
    BeliefNetworkProtocol,
    BeliefPropagationAnalyzerProtocol,
    CalibrationTrackerProtocol,
    CitationExtractorProtocol,
    CritiqueStoreProtocol,
    InsightExtractorProtocol,
    InsightStoreProtocol,
    PositionTrackerProtocol,
)


# =============================================================================
# Mock implementations for testing protocol compliance
# =============================================================================


class MockPositionTracker:
    """Minimal implementation of PositionTrackerProtocol."""

    def __init__(self) -> None:
        self._positions: dict[str, list[dict[str, Any]]] = {}
        self._history: dict[str, list[dict[str, Any]]] = {}

    def record_position(
        self,
        agent_name: str,
        claim_text: str,
        stance: str,
        confidence: float,
        context: str | None = None,
    ) -> None:
        """Record an agent's position on a claim."""
        position = {
            "claim": claim_text,
            "stance": stance,
            "confidence": confidence,
            "context": context,
        }
        if agent_name not in self._positions:
            self._positions[agent_name] = []
        self._positions[agent_name].append(position)

        if claim_text not in self._history:
            self._history[claim_text] = []
        self._history[claim_text].append({"agent": agent_name, **position})

    def get_positions(self, agent_name: str) -> list[dict[str, Any]]:
        """Get all positions recorded for an agent."""
        return self._positions.get(agent_name, [])

    def get_position_history(self, claim_text: str) -> list[dict[str, Any]]:
        """Get position history for a specific claim."""
        return self._history.get(claim_text, [])


class MockCalibrationTracker:
    """Minimal implementation of CalibrationTrackerProtocol."""

    def __init__(self) -> None:
        self._predictions: dict[str, dict[str, Any]] = {}
        self._agent_predictions: dict[str, list[str]] = {}
        self._id_counter = 0

    def record_prediction(
        self,
        agent_name: str,
        prediction: str,
        confidence: float,
        category: str | None = None,
    ) -> str:
        """Record a prediction for later resolution."""
        self._id_counter += 1
        pred_id = f"pred_{self._id_counter}"
        self._predictions[pred_id] = {
            "agent": agent_name,
            "prediction": prediction,
            "confidence": confidence,
            "category": category,
            "resolved": False,
            "outcome": None,
        }
        if agent_name not in self._agent_predictions:
            self._agent_predictions[agent_name] = []
        self._agent_predictions[agent_name].append(pred_id)
        return pred_id

    def resolve_prediction(
        self,
        prediction_id: str,
        outcome: bool,
    ) -> None:
        """Resolve a prediction as correct or incorrect."""
        if prediction_id in self._predictions:
            self._predictions[prediction_id]["resolved"] = True
            self._predictions[prediction_id]["outcome"] = outcome

    def get_calibration_score(self, agent_name: str) -> float | None:
        """Get an agent's calibration score."""
        pred_ids = self._agent_predictions.get(agent_name, [])
        resolved = [
            p for p in pred_ids
            if self._predictions[p]["resolved"]
        ]
        if not resolved:
            return None
        correct = sum(
            1 for p in resolved
            if self._predictions[p]["outcome"]
        )
        return correct / len(resolved)


class MockBeliefNetwork:
    """Minimal implementation of BeliefNetworkProtocol."""

    def __init__(self) -> None:
        self._claims: dict[str, dict[str, Any]] = {}
        self._supports: list[tuple[str, str, float]] = []
        self._attacks: list[tuple[str, str, float]] = []

    def add_claim(
        self,
        claim_id: str,
        text: str,
        confidence: float,
        agent: str | None = None,
    ) -> None:
        """Add a claim to the belief network."""
        self._claims[claim_id] = {
            "text": text,
            "confidence": confidence,
            "agent": agent,
        }

    def add_support(
        self,
        claim_id: str,
        supporting_claim_id: str,
        strength: float = 1.0,
    ) -> None:
        """Add a support relationship between claims."""
        self._supports.append((claim_id, supporting_claim_id, strength))

    def add_attack(
        self,
        claim_id: str,
        attacking_claim_id: str,
        strength: float = 1.0,
    ) -> None:
        """Add an attack relationship between claims."""
        self._attacks.append((claim_id, attacking_claim_id, strength))

    def propagate(self) -> dict[str, float]:
        """Propagate beliefs through the network."""
        # Simple mock: just return claim confidences
        return {cid: c["confidence"] for cid, c in self._claims.items()}


class MockBeliefPropagationAnalyzer:
    """Minimal implementation of BeliefPropagationAnalyzerProtocol."""

    def analyze_debate(
        self,
        messages: list[Any],
        network: BeliefNetworkProtocol | None = None,
    ) -> dict[str, Any]:
        """Analyze belief changes during a debate."""
        return {
            "message_count": len(messages),
            "network_provided": network is not None,
            "belief_changes": [],
        }


class MockCitationExtractor:
    """Minimal implementation of CitationExtractorProtocol."""

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract citations from text."""
        citations = []
        # Simple mock: detect URLs
        if "http" in text:
            citations.append({"type": "url", "text": "http_detected"})
        return citations

    def validate_citations(
        self,
        citations: list[dict[str, Any]],
        evidence_store: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Validate extracted citations."""
        return [
            {**c, "valid": True}
            for c in citations
        ]


class MockInsightExtractor:
    """Minimal implementation of InsightExtractorProtocol."""

    def extract_insights(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract insights from debate text."""
        if not text.strip():
            return []
        return [{"insight": "extracted", "text": text[:50]}]


class MockInsightStore:
    """Minimal implementation of InsightStoreProtocol."""

    def __init__(self) -> None:
        self._insights: dict[str, dict[str, Any]] = {}
        self._id_counter = 0

    def store_insight(
        self,
        insight: dict[str, Any],
        debate_id: str | None = None,
    ) -> str:
        """Store an insight and return its ID."""
        self._id_counter += 1
        insight_id = f"insight_{self._id_counter}"
        self._insights[insight_id] = {
            **insight,
            "debate_id": debate_id,
            "id": insight_id,
        }
        return insight_id

    def get_insights(
        self,
        debate_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve stored insights."""
        if debate_id is None:
            return list(self._insights.values())[:limit]
        return [
            i for i in self._insights.values()
            if i.get("debate_id") == debate_id
        ][:limit]

    def search_insights(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search insights by query."""
        query_lower = query.lower()
        results = []
        for insight in self._insights.values():
            text = str(insight.get("text", "")).lower()
            if query_lower in text:
                results.append(insight)
            if len(results) >= limit:
                break
        return results


class MockCritiqueStore:
    """Minimal implementation of CritiqueStoreProtocol."""

    def __init__(self) -> None:
        self._critiques: list[dict[str, Any]] = []

    def store_critique(
        self,
        critique: Any,
        debate_id: str | None = None,
    ) -> None:
        """Store a critique."""
        self._critiques.append({
            "critique": critique,
            "debate_id": debate_id,
        })

    def get_patterns(
        self,
        task: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get critique patterns relevant to a task."""
        # Simple mock: return all critiques up to limit
        return self._critiques[:limit]


class MockArgumentCartographer:
    """Minimal implementation of ArgumentCartographerProtocol."""

    def map_arguments(
        self,
        messages: list[Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Map argument structure from messages."""
        return {
            "nodes": [],
            "edges": [],
            "message_count": len(messages),
        }

    def visualize(
        self,
        argument_map: dict[str, Any],
        format: str = "json",
    ) -> Any:
        """Visualize the argument map."""
        if format == "json":
            return argument_map
        return str(argument_map)


# Non-compliant class (missing methods)
class InvalidPositionTracker:
    """Class that does not implement all protocol methods."""

    def record_position(self, agent_name: str, claim_text: str) -> None:
        """Incomplete signature."""
        pass


# =============================================================================
# Test Classes
# =============================================================================


class TestPositionTrackerProtocol:
    """Tests for PositionTrackerProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        tracker = MockPositionTracker()
        assert isinstance(tracker, PositionTrackerProtocol)

    def test_non_compliant_class_fails_check(self):
        """Non-compliant class should fail isinstance check."""
        invalid = InvalidPositionTracker()
        assert not isinstance(invalid, PositionTrackerProtocol)

    def test_record_position_basic(self):
        """Test recording a basic position."""
        tracker = MockPositionTracker()
        tracker.record_position(
            agent_name="claude",
            claim_text="Redis is best for caching",
            stance="support",
            confidence=0.85,
        )
        positions = tracker.get_positions("claude")
        assert len(positions) == 1
        assert positions[0]["claim"] == "Redis is best for caching"
        assert positions[0]["stance"] == "support"
        assert positions[0]["confidence"] == 0.85

    def test_record_position_with_context(self):
        """Test recording a position with context."""
        tracker = MockPositionTracker()
        tracker.record_position(
            agent_name="gpt4",
            claim_text="Use async IO",
            stance="support",
            confidence=0.9,
            context="performance optimization",
        )
        positions = tracker.get_positions("gpt4")
        assert positions[0]["context"] == "performance optimization"

    def test_get_positions_empty(self):
        """Test getting positions for unknown agent."""
        tracker = MockPositionTracker()
        assert tracker.get_positions("unknown") == []

    def test_get_position_history(self):
        """Test getting position history for a claim."""
        tracker = MockPositionTracker()
        tracker.record_position("claude", "claim1", "support", 0.8)
        tracker.record_position("gpt4", "claim1", "oppose", 0.7)

        history = tracker.get_position_history("claim1")
        assert len(history) == 2
        agents = {h["agent"] for h in history}
        assert agents == {"claude", "gpt4"}

    def test_get_position_history_empty(self):
        """Test getting history for unknown claim."""
        tracker = MockPositionTracker()
        assert tracker.get_position_history("unknown") == []


class TestCalibrationTrackerProtocol:
    """Tests for CalibrationTrackerProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        tracker = MockCalibrationTracker()
        assert isinstance(tracker, CalibrationTrackerProtocol)

    def test_record_prediction_returns_id(self):
        """Recording a prediction should return an ID."""
        tracker = MockCalibrationTracker()
        pred_id = tracker.record_prediction(
            agent_name="claude",
            prediction="The test will pass",
            confidence=0.8,
        )
        assert pred_id is not None
        assert isinstance(pred_id, str)

    def test_record_prediction_with_category(self):
        """Test recording a prediction with category."""
        tracker = MockCalibrationTracker()
        pred_id = tracker.record_prediction(
            agent_name="claude",
            prediction="Performance will improve",
            confidence=0.75,
            category="performance",
        )
        assert pred_id is not None

    def test_resolve_prediction(self):
        """Test resolving a prediction."""
        tracker = MockCalibrationTracker()
        pred_id = tracker.record_prediction("claude", "Success", 0.9)
        tracker.resolve_prediction(pred_id, True)
        score = tracker.get_calibration_score("claude")
        assert score == 1.0  # 100% correct

    def test_calibration_score_multiple_predictions(self):
        """Test calibration score with multiple predictions."""
        tracker = MockCalibrationTracker()
        p1 = tracker.record_prediction("claude", "Pred1", 0.8)
        p2 = tracker.record_prediction("claude", "Pred2", 0.6)
        p3 = tracker.record_prediction("claude", "Pred3", 0.9)

        tracker.resolve_prediction(p1, True)
        tracker.resolve_prediction(p2, False)
        tracker.resolve_prediction(p3, True)

        score = tracker.get_calibration_score("claude")
        assert score == pytest.approx(2/3, rel=0.01)

    def test_calibration_score_no_predictions(self):
        """Test calibration score for agent with no predictions."""
        tracker = MockCalibrationTracker()
        assert tracker.get_calibration_score("unknown") is None

    def test_calibration_score_unresolved_predictions(self):
        """Test calibration score when predictions are unresolved."""
        tracker = MockCalibrationTracker()
        tracker.record_prediction("claude", "Pending", 0.5)
        assert tracker.get_calibration_score("claude") is None


class TestBeliefNetworkProtocol:
    """Tests for BeliefNetworkProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        network = MockBeliefNetwork()
        assert isinstance(network, BeliefNetworkProtocol)

    def test_add_claim_basic(self):
        """Test adding a basic claim."""
        network = MockBeliefNetwork()
        network.add_claim("c1", "Redis is fast", 0.9)
        beliefs = network.propagate()
        assert "c1" in beliefs
        assert beliefs["c1"] == 0.9

    def test_add_claim_with_agent(self):
        """Test adding a claim with agent attribution."""
        network = MockBeliefNetwork()
        network.add_claim("c1", "Redis is fast", 0.85, agent="claude")
        beliefs = network.propagate()
        assert "c1" in beliefs

    def test_add_support_relationship(self):
        """Test adding support relationship."""
        network = MockBeliefNetwork()
        network.add_claim("c1", "Claim 1", 0.8)
        network.add_claim("c2", "Claim 2", 0.7)
        network.add_support("c1", "c2", strength=0.9)
        # Mock doesn't process relationships, but interface is tested
        assert len(network._supports) == 1

    def test_add_attack_relationship(self):
        """Test adding attack relationship."""
        network = MockBeliefNetwork()
        network.add_claim("c1", "Claim 1", 0.8)
        network.add_claim("c2", "Claim 2", 0.7)
        network.add_attack("c1", "c2", strength=0.5)
        assert len(network._attacks) == 1

    def test_propagate_returns_beliefs(self):
        """Test that propagate returns belief scores."""
        network = MockBeliefNetwork()
        network.add_claim("c1", "A", 0.5)
        network.add_claim("c2", "B", 0.6)
        beliefs = network.propagate()
        assert isinstance(beliefs, dict)
        assert len(beliefs) == 2


class TestBeliefPropagationAnalyzerProtocol:
    """Tests for BeliefPropagationAnalyzerProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        analyzer = MockBeliefPropagationAnalyzer()
        assert isinstance(analyzer, BeliefPropagationAnalyzerProtocol)

    def test_analyze_debate_basic(self):
        """Test analyzing a debate."""
        analyzer = MockBeliefPropagationAnalyzer()
        messages = [{"role": "proposer", "content": "Test"}]
        result = analyzer.analyze_debate(messages)
        assert result["message_count"] == 1
        assert "belief_changes" in result

    def test_analyze_debate_with_network(self):
        """Test analyzing a debate with provided network."""
        analyzer = MockBeliefPropagationAnalyzer()
        network = MockBeliefNetwork()
        messages = [{"role": "proposer", "content": "Test"}]
        result = analyzer.analyze_debate(messages, network=network)
        assert result["network_provided"] is True

    def test_analyze_debate_empty_messages(self):
        """Test analyzing with empty messages."""
        analyzer = MockBeliefPropagationAnalyzer()
        result = analyzer.analyze_debate([])
        assert result["message_count"] == 0


class TestCitationExtractorProtocol:
    """Tests for CitationExtractorProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        extractor = MockCitationExtractor()
        assert isinstance(extractor, CitationExtractorProtocol)

    def test_extract_with_url(self):
        """Test extracting citations with URL."""
        extractor = MockCitationExtractor()
        citations = extractor.extract("See https://example.com for details")
        assert len(citations) == 1
        assert citations[0]["type"] == "url"

    def test_extract_without_citations(self):
        """Test extracting from text without citations."""
        extractor = MockCitationExtractor()
        citations = extractor.extract("No citations here")
        assert citations == []

    def test_validate_citations(self):
        """Test validating citations."""
        extractor = MockCitationExtractor()
        citations = [{"type": "url", "text": "example.com"}]
        validated = extractor.validate_citations(citations)
        assert len(validated) == 1
        assert validated[0]["valid"] is True

    def test_validate_with_evidence_store(self):
        """Test validation with evidence store."""
        extractor = MockCitationExtractor()
        citations = [{"type": "url", "text": "example.com"}]
        validated = extractor.validate_citations(citations, evidence_store={})
        assert len(validated) == 1


class TestInsightExtractorProtocol:
    """Tests for InsightExtractorProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        extractor = MockInsightExtractor()
        assert isinstance(extractor, InsightExtractorProtocol)

    def test_extract_insights_basic(self):
        """Test extracting insights from text."""
        extractor = MockInsightExtractor()
        insights = extractor.extract_insights("Redis improves caching performance")
        assert len(insights) == 1
        assert "insight" in insights[0]

    def test_extract_insights_with_context(self):
        """Test extracting insights with context."""
        extractor = MockInsightExtractor()
        insights = extractor.extract_insights(
            "Redis is fast",
            context={"domain": "performance"},
        )
        assert len(insights) >= 1

    def test_extract_insights_empty_text(self):
        """Test extracting from empty text."""
        extractor = MockInsightExtractor()
        insights = extractor.extract_insights("")
        assert insights == []

    def test_extract_insights_whitespace_only(self):
        """Test extracting from whitespace-only text."""
        extractor = MockInsightExtractor()
        insights = extractor.extract_insights("   ")
        assert insights == []


class TestInsightStoreProtocol:
    """Tests for InsightStoreProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        store = MockInsightStore()
        assert isinstance(store, InsightStoreProtocol)

    def test_store_insight_returns_id(self):
        """Test storing an insight returns an ID."""
        store = MockInsightStore()
        insight_id = store.store_insight({"text": "Test insight"})
        assert insight_id is not None
        assert isinstance(insight_id, str)

    def test_store_insight_with_debate_id(self):
        """Test storing insight with debate ID."""
        store = MockInsightStore()
        insight_id = store.store_insight(
            {"text": "Test"},
            debate_id="debate_123",
        )
        insights = store.get_insights(debate_id="debate_123")
        assert len(insights) == 1

    def test_get_insights_all(self):
        """Test getting all insights."""
        store = MockInsightStore()
        store.store_insight({"text": "Insight 1"})
        store.store_insight({"text": "Insight 2"})
        insights = store.get_insights()
        assert len(insights) == 2

    def test_get_insights_with_limit(self):
        """Test getting insights with limit."""
        store = MockInsightStore()
        for i in range(5):
            store.store_insight({"text": f"Insight {i}"})
        insights = store.get_insights(limit=3)
        assert len(insights) == 3

    def test_get_insights_by_debate_id(self):
        """Test filtering insights by debate ID."""
        store = MockInsightStore()
        store.store_insight({"text": "A"}, debate_id="d1")
        store.store_insight({"text": "B"}, debate_id="d2")
        store.store_insight({"text": "C"}, debate_id="d1")

        insights = store.get_insights(debate_id="d1")
        assert len(insights) == 2

    def test_search_insights(self):
        """Test searching insights by query."""
        store = MockInsightStore()
        store.store_insight({"text": "Redis caching"})
        store.store_insight({"text": "PostgreSQL indexing"})
        store.store_insight({"text": "Redis performance"})

        results = store.search_insights("redis")
        assert len(results) == 2

    def test_search_insights_with_limit(self):
        """Test search with limit."""
        store = MockInsightStore()
        for i in range(5):
            store.store_insight({"text": f"Redis insight {i}"})

        results = store.search_insights("redis", limit=2)
        assert len(results) == 2

    def test_search_insights_no_match(self):
        """Test search with no matches."""
        store = MockInsightStore()
        store.store_insight({"text": "PostgreSQL"})
        results = store.search_insights("redis")
        assert results == []


class TestCritiqueStoreProtocol:
    """Tests for CritiqueStoreProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        store = MockCritiqueStore()
        assert isinstance(store, CritiqueStoreProtocol)

    def test_store_critique_basic(self):
        """Test storing a basic critique."""
        store = MockCritiqueStore()
        store.store_critique({"issues": ["Issue 1"]})
        patterns = store.get_patterns("test task")
        assert len(patterns) == 1

    def test_store_critique_with_debate_id(self):
        """Test storing critique with debate ID."""
        store = MockCritiqueStore()
        store.store_critique({"issues": []}, debate_id="debate_123")
        assert len(store._critiques) == 1

    def test_get_patterns_with_limit(self):
        """Test getting patterns with limit."""
        store = MockCritiqueStore()
        for i in range(5):
            store.store_critique({"id": i})
        patterns = store.get_patterns("task", limit=3)
        assert len(patterns) == 3

    def test_get_patterns_empty_store(self):
        """Test getting patterns from empty store."""
        store = MockCritiqueStore()
        patterns = store.get_patterns("task")
        assert patterns == []


class TestArgumentCartographerProtocol:
    """Tests for ArgumentCartographerProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        cartographer = MockArgumentCartographer()
        assert isinstance(cartographer, ArgumentCartographerProtocol)

    def test_map_arguments_basic(self):
        """Test mapping arguments from messages."""
        cartographer = MockArgumentCartographer()
        messages = [{"role": "proposer", "content": "Argument 1"}]
        result = cartographer.map_arguments(messages)
        assert "nodes" in result
        assert "edges" in result
        assert result["message_count"] == 1

    def test_map_arguments_with_context(self):
        """Test mapping arguments with context."""
        cartographer = MockArgumentCartographer()
        messages = [{"content": "Test"}]
        context = {"domain": "security"}
        result = cartographer.map_arguments(messages, context=context)
        assert isinstance(result, dict)

    def test_map_arguments_empty_messages(self):
        """Test mapping with empty messages."""
        cartographer = MockArgumentCartographer()
        result = cartographer.map_arguments([])
        assert result["message_count"] == 0

    def test_visualize_json_format(self):
        """Test visualizing in JSON format."""
        cartographer = MockArgumentCartographer()
        arg_map = {"nodes": ["a", "b"], "edges": []}
        result = cartographer.visualize(arg_map, format="json")
        assert result == arg_map

    def test_visualize_string_format(self):
        """Test visualizing in string format."""
        cartographer = MockArgumentCartographer()
        arg_map = {"nodes": ["a"], "edges": []}
        result = cartographer.visualize(arg_map, format="text")
        assert isinstance(result, str)


class TestProtocolExports:
    """Tests for module exports."""

    def test_all_protocols_exported(self):
        """Verify all protocols are in __all__."""
        from aragora.debate import interfaces

        expected = {
            "PositionTrackerProtocol",
            "CalibrationTrackerProtocol",
            "BeliefNetworkProtocol",
            "BeliefPropagationAnalyzerProtocol",
            "CitationExtractorProtocol",
            "InsightExtractorProtocol",
            "InsightStoreProtocol",
            "CritiqueStoreProtocol",
            "ArgumentCartographerProtocol",
        }
        assert set(interfaces.__all__) == expected

    def test_protocols_are_protocol_type(self):
        """Verify all exported items are Protocol classes."""
        from typing import Protocol

        protocols = [
            PositionTrackerProtocol,
            CalibrationTrackerProtocol,
            BeliefNetworkProtocol,
            BeliefPropagationAnalyzerProtocol,
            CitationExtractorProtocol,
            InsightExtractorProtocol,
            InsightStoreProtocol,
            CritiqueStoreProtocol,
            ArgumentCartographerProtocol,
        ]
        for proto in protocols:
            # Check that it's a subclass of Protocol
            assert issubclass(proto, Protocol)
