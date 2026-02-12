"""
Tests for InsightsAdapter - Bridges Insights/Trickster to the Knowledge Mound.

Tests cover:
- Insight storage and retrieval (CRUD operations)
- Flip event storage and retrieval
- Pattern storage and aggregation
- Search and filtering functionality
- Conversion to KnowledgeItem
- Reverse flow operations (KM -> InsightStore/FlipDetector)
- Statistics and edge cases
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# Mock Classes for Testing
# ============================================================================


class MockInsightType:
    """Mock InsightType enum for testing."""

    def __init__(self, value: str):
        self.value = value


@dataclass
class MockInsight:
    """Mock Insight for testing without importing the real class."""

    id: str
    type: MockInsightType
    title: str
    description: str
    confidence: float
    debate_id: str
    agents_involved: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)


@dataclass
class MockFlipEvent:
    """Mock FlipEvent for testing without importing the real class."""

    id: str
    agent_name: str
    original_claim: str
    new_claim: str
    original_confidence: float
    new_confidence: float
    original_debate_id: str
    new_debate_id: str
    original_position_id: str = ""
    new_position_id: str = ""
    similarity_score: float = 0.8
    flip_type: str = "contradiction"
    domain: str | None = None
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MockDebateInsights:
    """Mock DebateInsights for testing."""

    debate_id: str
    _insights: list = field(default_factory=list)

    def all_insights(self):
        return self._insights


# ============================================================================
# InsightsAdapter Initialization Tests
# ============================================================================


class TestInsightsAdapterInit:
    """Tests for InsightsAdapter initialization."""

    def test_init_default(self):
        """Should initialize with default values."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        assert adapter._insight_store is None
        assert adapter._flip_detector is None
        assert adapter._enable_dual_write is False
        assert adapter._event_callback is None
        assert adapter._insights == {}
        assert adapter._flips == {}
        assert adapter._patterns == {}

    def test_init_with_stores(self):
        """Should initialize with insight store and flip detector."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        mock_store = MagicMock()
        mock_detector = MagicMock()

        adapter = InsightsAdapter(
            insight_store=mock_store,
            flip_detector=mock_detector,
            enable_dual_write=True,
        )

        assert adapter.insight_store is mock_store
        assert adapter.flip_detector is mock_detector
        assert adapter._enable_dual_write is True

    def test_init_with_event_callback(self):
        """Should initialize with event callback."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        callback = MagicMock()
        adapter = InsightsAdapter(event_callback=callback)

        assert adapter._event_callback is callback


class TestEventCallbackFunctionality:
    """Tests for event callback functionality."""

    def test_set_event_callback(self):
        """Should set event callback."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        callback = MagicMock()

        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback

    def test_emit_event_calls_callback(self):
        """Should emit event via callback."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        callback = MagicMock()
        adapter = InsightsAdapter(event_callback=callback)

        adapter._emit_event("test_event", {"key": "value"})

        callback.assert_called_once_with("test_event", {"key": "value"})

    def test_emit_event_handles_callback_error(self):
        """Should handle callback errors gracefully."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        callback = MagicMock(side_effect=Exception("Callback error"))
        adapter = InsightsAdapter(event_callback=callback)

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})

    def test_emit_event_without_callback(self):
        """Should handle missing callback gracefully."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Should not raise
        adapter._emit_event("test_event", {"key": "value"})


# ============================================================================
# Insight Storage Tests
# ============================================================================


class TestStoreInsight:
    """Tests for storing insights."""

    def test_store_insight_above_threshold(self):
        """Should store insight when confidence is above threshold."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight = MockInsight(
            id="insight-1",
            type=MockInsightType("consensus"),
            title="Test Insight",
            description="Test description",
            confidence=0.85,
            debate_id="debate-1",
            agents_involved=["agent1", "agent2"],
        )

        result = adapter.store_insight(insight)

        assert result == "in_insight-1"
        assert "in_insight-1" in adapter._insights
        assert adapter._insights["in_insight-1"]["title"] == "Test Insight"

    def test_store_insight_below_threshold(self):
        """Should not store insight when confidence is below threshold."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight = MockInsight(
            id="insight-2",
            type=MockInsightType("pattern"),
            title="Low Confidence Insight",
            description="Test",
            confidence=0.5,  # Below default threshold of 0.7
            debate_id="debate-1",
        )

        result = adapter.store_insight(insight)

        assert result is None
        assert "in_insight-2" not in adapter._insights

    def test_store_insight_custom_threshold(self):
        """Should use custom confidence threshold."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight = MockInsight(
            id="insight-3",
            type=MockInsightType("dissent"),
            title="Custom Threshold Insight",
            description="Test",
            confidence=0.5,
            debate_id="debate-1",
        )

        result = adapter.store_insight(insight, min_confidence=0.4)

        assert result == "in_insight-3"
        assert "in_insight-3" in adapter._insights

    def test_store_insight_updates_indices(self):
        """Should update debate and type indices."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight = MockInsight(
            id="insight-4",
            type=MockInsightType("pattern"),
            title="Indexed Insight",
            description="Test",
            confidence=0.9,
            debate_id="debate-100",
        )

        adapter.store_insight(insight)

        assert "in_insight-4" in adapter._debate_insights.get("debate-100", [])
        assert "in_insight-4" in adapter._type_insights.get("pattern", [])


class TestStoreDebateInsights:
    """Tests for storing batch debate insights."""

    def test_store_debate_insights(self):
        """Should store all insights from a debate."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insights = [
            MockInsight(
                id=f"insight-{i}",
                type=MockInsightType("consensus"),
                title=f"Insight {i}",
                description="Test",
                confidence=0.85,
                debate_id="debate-batch-1",
            )
            for i in range(3)
        ]
        debate_insights = MockDebateInsights(debate_id="debate-batch-1", _insights=insights)

        result = adapter.store_debate_insights(debate_insights)

        assert len(result) == 3
        assert all(r.startswith("in_") for r in result)

    def test_store_debate_insights_filters_low_confidence(self):
        """Should filter out low confidence insights from batch."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insights = [
            MockInsight(
                id="high-conf",
                type=MockInsightType("consensus"),
                title="High Confidence",
                description="Test",
                confidence=0.9,
                debate_id="debate-batch-2",
            ),
            MockInsight(
                id="low-conf",
                type=MockInsightType("pattern"),
                title="Low Confidence",
                description="Test",
                confidence=0.3,
                debate_id="debate-batch-2",
            ),
        ]
        debate_insights = MockDebateInsights(debate_id="debate-batch-2", _insights=insights)

        result = adapter.store_debate_insights(debate_insights)

        assert len(result) == 1
        assert result[0] == "in_high-conf"


# ============================================================================
# Flip Event Storage Tests
# ============================================================================


class TestStoreFlip:
    """Tests for storing flip events."""

    def test_store_flip(self):
        """Should store a flip event."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flip = MockFlipEvent(
            id="flip-1",
            agent_name="claude",
            original_claim="X is good",
            new_claim="X is bad",
            original_confidence=0.8,
            new_confidence=0.9,
            original_debate_id="debate-1",
            new_debate_id="debate-2",
            similarity_score=0.85,
            flip_type="contradiction",
            domain="technology",
        )

        result = adapter.store_flip(flip)

        assert result == "fl_flip-1"
        assert "fl_flip-1" in adapter._flips
        assert adapter._flips["fl_flip-1"]["agent_name"] == "claude"

    def test_store_flip_updates_agent_index(self):
        """Should update agent flips index."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flip = MockFlipEvent(
            id="flip-2",
            agent_name="gpt-4",
            original_claim="Claim A",
            new_claim="Claim B",
            original_confidence=0.7,
            new_confidence=0.8,
            original_debate_id="d1",
            new_debate_id="d2",
        )

        adapter.store_flip(flip)

        assert "fl_flip-2" in adapter._agent_flips.get("gpt-4", [])

    def test_store_flip_updates_domain_index(self):
        """Should update domain flips index."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flip = MockFlipEvent(
            id="flip-3",
            agent_name="gemini",
            original_claim="Claim",
            new_claim="Different claim",
            original_confidence=0.6,
            new_confidence=0.7,
            original_debate_id="d1",
            new_debate_id="d2",
            domain="security",
        )

        adapter.store_flip(flip)

        assert "fl_flip-3" in adapter._domain_flips.get("security", [])

    def test_store_flip_without_domain(self):
        """Should handle flip without domain."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flip = MockFlipEvent(
            id="flip-4",
            agent_name="mistral",
            original_claim="Claim",
            new_claim="Different",
            original_confidence=0.5,
            new_confidence=0.6,
            original_debate_id="d1",
            new_debate_id="d2",
            domain=None,
        )

        result = adapter.store_flip(flip)

        assert result == "fl_flip-4"
        assert "fl_flip-4" in adapter._flips


class TestStoreFlipsBatch:
    """Tests for batch flip storage."""

    def test_store_flips_batch(self):
        """Should store multiple flip events."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flips = [
            MockFlipEvent(
                id=f"flip-batch-{i}",
                agent_name="claude",
                original_claim=f"Claim {i}",
                new_claim=f"New claim {i}",
                original_confidence=0.7,
                new_confidence=0.8,
                original_debate_id="d1",
                new_debate_id="d2",
            )
            for i in range(5)
        ]

        result = adapter.store_flips_batch(flips)

        assert len(result) == 5
        assert all(r.startswith("fl_") for r in result)


# ============================================================================
# Pattern Storage Tests
# ============================================================================


class TestStorePattern:
    """Tests for storing patterns."""

    def test_store_pattern_above_threshold(self):
        """Should store pattern with sufficient occurrences."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.store_pattern(
            category="security",
            pattern_text="Agents disagree on encryption strength",
            occurrence_count=5,
            avg_severity=0.7,
            debate_ids=["d1", "d2", "d3"],
        )

        assert result is not None
        assert result.startswith("pt_")
        assert result in adapter._patterns

    def test_store_pattern_below_threshold(self):
        """Should not store pattern with too few occurrences."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.store_pattern(
            category="performance",
            pattern_text="Rare pattern",
            occurrence_count=2,  # Below default threshold of 3
        )

        assert result is None

    def test_store_pattern_updates_existing(self):
        """Should update existing pattern with additional occurrences."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Store initial pattern
        result1 = adapter.store_pattern(
            category="testing",
            pattern_text="Missing edge case tests",
            occurrence_count=4,
            debate_ids=["d1", "d2"],
        )

        # Store same pattern again (should update)
        result2 = adapter.store_pattern(
            category="testing",
            pattern_text="Missing edge case tests",
            occurrence_count=3,
            debate_ids=["d3", "d4"],
        )

        assert result1 == result2
        pattern = adapter._patterns[result1]
        assert pattern["occurrence_count"] == 7  # 4 + 3
        assert len(pattern["debate_ids"]) == 4  # Unique debate IDs


# ============================================================================
# Retrieval Tests
# ============================================================================


class TestGetInsight:
    """Tests for getting insights by ID."""

    def test_get_insight_with_prefix(self):
        """Should get insight by full ID with prefix."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight = MockInsight(
            id="get-test-1",
            type=MockInsightType("consensus"),
            title="Test",
            description="Test",
            confidence=0.9,
            debate_id="d1",
        )
        adapter.store_insight(insight)

        result = adapter.get_insight("in_get-test-1")

        assert result is not None
        assert result["original_id"] == "get-test-1"

    def test_get_insight_without_prefix(self):
        """Should get insight by ID without prefix."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight = MockInsight(
            id="get-test-2",
            type=MockInsightType("pattern"),
            title="Test",
            description="Test",
            confidence=0.9,
            debate_id="d1",
        )
        adapter.store_insight(insight)

        result = adapter.get_insight("get-test-2")

        assert result is not None
        assert result["original_id"] == "get-test-2"

    def test_get_insight_not_found(self):
        """Should return None for non-existent insight."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.get_insight("non-existent")

        assert result is None


class TestGetFlip:
    """Tests for getting flip events by ID."""

    def test_get_flip_with_prefix(self):
        """Should get flip by full ID with prefix."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flip = MockFlipEvent(
            id="get-flip-1",
            agent_name="claude",
            original_claim="X",
            new_claim="Y",
            original_confidence=0.7,
            new_confidence=0.8,
            original_debate_id="d1",
            new_debate_id="d2",
        )
        adapter.store_flip(flip)

        result = adapter.get_flip("fl_get-flip-1")

        assert result is not None
        assert result["original_id"] == "get-flip-1"

    def test_get_flip_without_prefix(self):
        """Should get flip by ID without prefix."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flip = MockFlipEvent(
            id="get-flip-2",
            agent_name="gpt-4",
            original_claim="A",
            new_claim="B",
            original_confidence=0.6,
            new_confidence=0.7,
            original_debate_id="d1",
            new_debate_id="d2",
        )
        adapter.store_flip(flip)

        result = adapter.get_flip("get-flip-2")

        assert result is not None

    def test_get_flip_not_found(self):
        """Should return None for non-existent flip."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.get_flip("non-existent")

        assert result is None


# ============================================================================
# Search and Filter Tests
# ============================================================================


class TestSearchSimilarInsights:
    """Tests for searching similar insights."""

    def test_search_finds_matching_insights(self):
        """Should find insights matching query keywords."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Store some insights
        insights = [
            MockInsight(
                id="search-1",
                type=MockInsightType("consensus"),
                title="Security vulnerability found",
                description="SQL injection in login form",
                confidence=0.9,
                debate_id="d1",
            ),
            MockInsight(
                id="search-2",
                type=MockInsightType("pattern"),
                title="Performance improvement",
                description="Caching reduces latency",
                confidence=0.85,
                debate_id="d2",
            ),
            MockInsight(
                id="search-3",
                type=MockInsightType("consensus"),
                title="Security best practices",
                description="Use prepared statements",
                confidence=0.8,
                debate_id="d3",
            ),
        ]
        for insight in insights:
            adapter.store_insight(insight)

        result = adapter.search_similar_insights("security vulnerability")

        assert len(result) >= 1
        assert any("security" in r["title"].lower() for r in result)

    def test_search_respects_limit(self):
        """Should respect result limit."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(10):
            insight = MockInsight(
                id=f"limit-{i}",
                type=MockInsightType("pattern"),
                title=f"Pattern matching keyword {i}",
                description="Test keyword description",
                confidence=0.9,
                debate_id=f"d{i}",
            )
            adapter.store_insight(insight)

        result = adapter.search_similar_insights("keyword pattern", limit=3)

        assert len(result) <= 3

    def test_search_filters_by_type(self):
        """Should filter by insight type."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        insights = [
            MockInsight(
                id="type-1",
                type=MockInsightType("consensus"),
                title="Consensus on testing",
                description="All agree on testing",
                confidence=0.9,
                debate_id="d1",
            ),
            MockInsight(
                id="type-2",
                type=MockInsightType("pattern"),
                title="Testing pattern found",
                description="Repeated testing issue",
                confidence=0.9,
                debate_id="d2",
            ),
        ]
        for insight in insights:
            adapter.store_insight(insight)

        result = adapter.search_similar_insights("testing", insight_type="consensus")

        assert len(result) == 1
        assert result[0]["type"] == "consensus"

    def test_search_filters_by_confidence(self):
        """Should filter by minimum confidence."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        insights = [
            MockInsight(
                id="conf-1",
                type=MockInsightType("pattern"),
                title="High confidence keyword",
                description="Test",
                confidence=0.95,
                debate_id="d1",
            ),
            MockInsight(
                id="conf-2",
                type=MockInsightType("pattern"),
                title="Low confidence keyword",
                description="Test",
                confidence=0.75,
                debate_id="d2",
            ),
        ]
        for insight in insights:
            adapter.store_insight(insight)

        result = adapter.search_similar_insights("keyword", min_confidence=0.9)

        assert len(result) == 1
        assert result[0]["confidence"] >= 0.9

    def test_search_empty_query(self):
        """Should handle empty query gracefully."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight = MockInsight(
            id="empty-query",
            type=MockInsightType("consensus"),
            title="Test",
            description="Test",
            confidence=0.9,
            debate_id="d1",
        )
        adapter.store_insight(insight)

        result = adapter.search_similar_insights("")

        # Empty query has no matching words
        assert len(result) == 0


class TestGetInsightsByType:
    """Tests for getting insights by type."""

    def test_get_insights_by_type(self):
        """Should get all insights of a specific type."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(5):
            insight = MockInsight(
                id=f"by-type-{i}",
                type=MockInsightType("dissent"),
                title=f"Dissent {i}",
                description="Test",
                confidence=0.9,
                debate_id=f"d{i}",
            )
            adapter.store_insight(insight)

        result = adapter.get_insights_by_type("dissent")

        assert len(result) == 5
        assert all(r["type"] == "dissent" for r in result)

    def test_get_insights_by_type_respects_limit(self):
        """Should respect result limit."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(10):
            insight = MockInsight(
                id=f"type-limit-{i}",
                type=MockInsightType("pattern"),
                title=f"Pattern {i}",
                description="Test",
                confidence=0.9,
                debate_id=f"d{i}",
            )
            adapter.store_insight(insight)

        result = adapter.get_insights_by_type("pattern", limit=3)

        assert len(result) == 3

    def test_get_insights_by_type_unknown_type(self):
        """Should return empty list for unknown type."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.get_insights_by_type("unknown_type")

        assert result == []


class TestGetDebateInsights:
    """Tests for getting insights by debate ID."""

    def test_get_debate_insights(self):
        """Should get all insights from a specific debate."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(3):
            insight = MockInsight(
                id=f"debate-ins-{i}",
                type=MockInsightType("consensus"),
                title=f"Insight {i}",
                description="Test",
                confidence=0.9,
                debate_id="target-debate",
            )
            adapter.store_insight(insight)

        result = adapter.get_debate_insights("target-debate")

        assert len(result) == 3

    def test_get_debate_insights_filters_by_confidence(self):
        """Should filter by minimum confidence."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        insights = [
            MockInsight(
                id="deb-high",
                type=MockInsightType("consensus"),
                title="High",
                description="Test",
                confidence=0.95,
                debate_id="conf-debate",
            ),
            MockInsight(
                id="deb-low",
                type=MockInsightType("pattern"),
                title="Low",
                description="Test",
                confidence=0.75,
                debate_id="conf-debate",
            ),
        ]
        for insight in insights:
            adapter.store_insight(insight)

        result = adapter.get_debate_insights("conf-debate", min_confidence=0.9)

        assert len(result) == 1
        assert result[0]["confidence"] >= 0.9

    def test_get_debate_insights_unknown_debate(self):
        """Should return empty list for unknown debate."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.get_debate_insights("unknown-debate")

        assert result == []


class TestGetAgentFlipHistory:
    """Tests for getting agent flip history."""

    def test_get_agent_flip_history(self):
        """Should get flip history for an agent."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(5):
            flip = MockFlipEvent(
                id=f"agent-flip-{i}",
                agent_name="target-agent",
                original_claim=f"Claim {i}",
                new_claim=f"New claim {i}",
                original_confidence=0.7,
                new_confidence=0.8,
                original_debate_id=f"d{i}",
                new_debate_id=f"d{i + 1}",
            )
            adapter.store_flip(flip)

        result = adapter.get_agent_flip_history("target-agent")

        assert len(result) == 5

    def test_get_agent_flip_history_respects_limit(self):
        """Should respect result limit."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(10):
            flip = MockFlipEvent(
                id=f"limit-flip-{i}",
                agent_name="limit-agent",
                original_claim=f"Claim {i}",
                new_claim=f"New {i}",
                original_confidence=0.5,
                new_confidence=0.6,
                original_debate_id=f"d{i}",
                new_debate_id=f"d{i + 1}",
            )
            adapter.store_flip(flip)

        result = adapter.get_agent_flip_history("limit-agent", limit=3)

        assert len(result) == 3

    def test_get_agent_flip_history_filters_by_type(self):
        """Should filter by flip type."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        flips = [
            MockFlipEvent(
                id="type-flip-1",
                agent_name="typed-agent",
                original_claim="A",
                new_claim="B",
                original_confidence=0.7,
                new_confidence=0.8,
                original_debate_id="d1",
                new_debate_id="d2",
                flip_type="contradiction",
            ),
            MockFlipEvent(
                id="type-flip-2",
                agent_name="typed-agent",
                original_claim="X",
                new_claim="Y",
                original_confidence=0.6,
                new_confidence=0.7,
                original_debate_id="d3",
                new_debate_id="d4",
                flip_type="refinement",
            ),
        ]
        for flip in flips:
            adapter.store_flip(flip)

        result = adapter.get_agent_flip_history("typed-agent", flip_type="contradiction")

        assert len(result) == 1
        assert result[0]["flip_type"] == "contradiction"

    def test_get_agent_flip_history_unknown_agent(self):
        """Should return empty list for unknown agent."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.get_agent_flip_history("unknown-agent")

        assert result == []


class TestGetDomainFlips:
    """Tests for getting flips by domain."""

    def test_get_domain_flips(self):
        """Should get flips for a domain."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(3):
            flip = MockFlipEvent(
                id=f"domain-flip-{i}",
                agent_name=f"agent-{i}",
                original_claim="Claim",
                new_claim="New claim",
                original_confidence=0.7,
                new_confidence=0.8,
                original_debate_id="d1",
                new_debate_id="d2",
                domain="target-domain",
            )
            adapter.store_flip(flip)

        result = adapter.get_domain_flips("target-domain")

        assert len(result) == 3

    def test_get_domain_flips_unknown_domain(self):
        """Should return empty list for unknown domain."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = adapter.get_domain_flips("unknown-domain")

        assert result == []


class TestGetCommonPatterns:
    """Tests for getting common patterns."""

    def test_get_common_patterns(self):
        """Should get patterns above occurrence threshold."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        adapter.store_pattern("category1", "Pattern A", 5)
        adapter.store_pattern("category1", "Pattern B", 10)
        adapter.store_pattern("category2", "Pattern C", 3)

        result = adapter.get_common_patterns(min_occurrences=3)

        assert len(result) == 3

    def test_get_common_patterns_filters_by_category(self):
        """Should filter by category."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        adapter.store_pattern("security", "Security pattern", 5)
        adapter.store_pattern("performance", "Performance pattern", 5)

        result = adapter.get_common_patterns(category="security")

        assert len(result) == 1
        assert result[0]["category"] == "security"

    def test_get_common_patterns_sorted_by_occurrence(self):
        """Should sort by occurrence count descending."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        adapter.store_pattern("cat", "Low count", 4)
        adapter.store_pattern("cat", "High count", 10)
        adapter.store_pattern("cat", "Medium count", 6)

        result = adapter.get_common_patterns()

        assert result[0]["occurrence_count"] == 10
        assert result[1]["occurrence_count"] == 6
        assert result[2]["occurrence_count"] == 4


# ============================================================================
# Conversion Tests
# ============================================================================


class TestToKnowledgeItem:
    """Tests for converting insights to KnowledgeItem."""

    def test_to_knowledge_item_high_confidence(self):
        """Should convert high confidence insight correctly."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter
        from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource

        adapter = InsightsAdapter()
        insight_dict = {
            "id": "ki-test-1",
            "original_id": "test-1",
            "type": "consensus",
            "title": "Test Title",
            "description": "Test Description",
            "confidence": 0.95,
            "debate_id": "d1",
            "agents_involved": ["agent1"],
            "evidence": ["ev1"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        result = adapter.to_knowledge_item(insight_dict)

        assert result.id == "ki-test-1"
        assert result.source == KnowledgeSource.INSIGHT
        assert result.confidence == ConfidenceLevel.VERIFIED
        assert "Test Title" in result.content

    def test_to_knowledge_item_various_confidence_levels(self):
        """Should map confidence values to correct levels."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter
        from aragora.knowledge.unified.types import ConfidenceLevel

        adapter = InsightsAdapter()

        test_cases = [
            (0.95, ConfidenceLevel.VERIFIED),
            (0.75, ConfidenceLevel.HIGH),
            (0.55, ConfidenceLevel.MEDIUM),
            (0.35, ConfidenceLevel.LOW),
            (0.1, ConfidenceLevel.UNVERIFIED),
        ]

        for conf_value, expected_level in test_cases:
            insight_dict = {
                "id": f"ki-conf-{conf_value}",
                "title": "Test",
                "description": "Test",
                "confidence": conf_value,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            result = adapter.to_knowledge_item(insight_dict)
            assert result.confidence == expected_level, f"Failed for confidence {conf_value}"


class TestFlipToKnowledgeItem:
    """Tests for converting flips to KnowledgeItem."""

    def test_flip_to_knowledge_item(self):
        """Should convert flip to KnowledgeItem."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter
        from aragora.knowledge.unified.types import KnowledgeSource

        adapter = InsightsAdapter()
        flip_dict = {
            "id": "fl_flip-ki-1",
            "original_id": "flip-ki-1",
            "agent_name": "claude",
            "original_claim": "X is true",
            "new_claim": "X is false",
            "similarity_score": 0.85,
            "flip_type": "contradiction",
            "domain": "technology",
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

        result = adapter.flip_to_knowledge_item(flip_dict)

        assert result.id == "fl_flip-ki-1"
        assert result.source == KnowledgeSource.FLIP
        assert "claude" in result.content
        assert result.metadata["flip_type"] == "contradiction"


# ============================================================================
# Statistics Tests
# ============================================================================


class TestGetStats:
    """Tests for getting adapter statistics."""

    def test_get_stats_empty(self):
        """Should return stats for empty adapter."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        stats = adapter.get_stats()

        assert stats["total_insights"] == 0
        assert stats["total_flips"] == 0
        assert stats["total_patterns"] == 0

    def test_get_stats_with_data(self):
        """Should return accurate stats with data."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Add insights
        for i in range(3):
            insight = MockInsight(
                id=f"stat-insight-{i}",
                type=MockInsightType("consensus"),
                title="Test",
                description="Test",
                confidence=0.9,
                debate_id="d1",
            )
            adapter.store_insight(insight)

        # Add flips
        for i in range(2):
            flip = MockFlipEvent(
                id=f"stat-flip-{i}",
                agent_name="agent1",
                original_claim="A",
                new_claim="B",
                original_confidence=0.7,
                new_confidence=0.8,
                original_debate_id="d1",
                new_debate_id="d2",
                flip_type="contradiction",
            )
            adapter.store_flip(flip)

        # Add pattern
        adapter.store_pattern("category", "Pattern", 5)

        stats = adapter.get_stats()

        assert stats["total_insights"] == 3
        assert stats["total_flips"] == 2
        assert stats["total_patterns"] == 1
        assert stats["flip_types"]["contradiction"] == 2


# ============================================================================
# Reverse Flow Tests
# ============================================================================


class TestRecordOutcome:
    """Tests for recording flip outcomes."""

    def test_record_outcome(self):
        """Should record flip outcome."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        adapter.record_outcome(
            flip_id="fl_test-1",
            debate_id="d1",
            was_accurate=True,
            confidence=0.8,
        )

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_size"] == 1


@pytest.mark.asyncio
class TestUpdateFlipThresholdsFromKM:
    """Tests for updating flip thresholds from KM patterns."""

    async def test_update_thresholds_with_patterns(self):
        """Should update thresholds based on KM patterns."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Create KM items with similarity and accuracy data
        km_items = [
            {"metadata": {"similarity_score": 0.85, "was_accurate": True}} for _ in range(10)
        ]

        result = await adapter.update_flip_thresholds_from_km(km_items)

        assert result.patterns_analyzed == 10
        assert result.old_similarity_threshold == 0.7

    async def test_update_thresholds_keeps_when_no_data(self):
        """Should keep thresholds when insufficient data."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = await adapter.update_flip_thresholds_from_km([])

        assert result.recommendation == "keep"
        assert result.patterns_analyzed == 0


@pytest.mark.asyncio
class TestGetAgentFlipBaselines:
    """Tests for getting agent flip baselines."""

    async def test_get_baseline_for_agent(self):
        """Should compute baseline for an agent."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Add some flips for the agent
        for i in range(5):
            flip = MockFlipEvent(
                id=f"baseline-flip-{i}",
                agent_name="baseline-agent",
                original_claim=f"Claim {i}",
                new_claim=f"New claim {i}",
                original_confidence=0.7,
                new_confidence=0.8,
                original_debate_id=f"d{i}",
                new_debate_id=f"d{i + 1}",
                flip_type="contradiction" if i % 2 == 0 else "refinement",
            )
            adapter.store_flip(flip)

        result = await adapter.get_agent_flip_baselines("baseline-agent")

        assert result.agent_name == "baseline-agent"
        assert result.sample_count == 5
        assert "contradiction" in result.flip_type_distribution

    async def test_baseline_caching(self):
        """Should cache baselines."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        flip = MockFlipEvent(
            id="cache-flip",
            agent_name="cached-agent",
            original_claim="A",
            new_claim="B",
            original_confidence=0.7,
            new_confidence=0.8,
            original_debate_id="d1",
            new_debate_id="d2",
        )
        adapter.store_flip(flip)

        # First call
        result1 = await adapter.get_agent_flip_baselines("cached-agent")
        # Second call (should use cache)
        result2 = await adapter.get_agent_flip_baselines("cached-agent")

        assert result1 == result2


@pytest.mark.asyncio
class TestValidateFlipFromKM:
    """Tests for validating flips using KM patterns."""

    async def test_validate_existing_flip(self):
        """Should validate an existing flip."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        flip = MockFlipEvent(
            id="validate-flip",
            agent_name="validate-agent",
            original_claim="X",
            new_claim="Y",
            original_confidence=0.7,
            new_confidence=0.8,
            original_debate_id="d1",
            new_debate_id="d2",
            flip_type="contradiction",
            similarity_score=0.85,
        )
        adapter.store_flip(flip)

        km_patterns = [{"metadata": {"flip_type": "contradiction", "relationship": "supports"}}]

        result = await adapter.validate_flip_from_km("fl_validate-flip", km_patterns)

        assert result.flip_id == "fl_validate-flip"
        assert result.km_confidence > 0

    async def test_validate_nonexistent_flip(self):
        """Should handle validation of non-existent flip."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        result = await adapter.validate_flip_from_km("non-existent", [])

        assert result.km_confidence == 0.0
        assert result.metadata.get("error") == "flip_not_found"


@pytest.mark.asyncio
class TestApplyKMValidation:
    """Tests for applying KM validations."""

    async def test_apply_validation_to_flip(self):
        """Should apply validation to existing flip."""
        from aragora.knowledge.mound.adapters.insights_adapter import (
            InsightsAdapter,
            KMFlipValidation,
        )

        adapter = InsightsAdapter()

        flip = MockFlipEvent(
            id="apply-flip",
            agent_name="apply-agent",
            original_claim="A",
            new_claim="B",
            original_confidence=0.7,
            new_confidence=0.8,
            original_debate_id="d1",
            new_debate_id="d2",
        )
        adapter.store_flip(flip)

        validation = KMFlipValidation(
            flip_id="fl_apply-flip",
            km_confidence=0.85,
            is_expected=True,
            recommendation="keep",
        )

        result = await adapter.apply_km_validation(validation)

        assert result is True
        flip_data = adapter.get_flip("apply-flip")
        assert flip_data["km_validated"] is True

    async def test_apply_validation_to_nonexistent_flip(self):
        """Should return False for non-existent flip."""
        from aragora.knowledge.mound.adapters.insights_adapter import (
            InsightsAdapter,
            KMFlipValidation,
        )

        adapter = InsightsAdapter()

        validation = KMFlipValidation(
            flip_id="non-existent",
            km_confidence=0.5,
        )

        result = await adapter.apply_km_validation(validation)

        assert result is False


@pytest.mark.asyncio
class TestSyncValidationsFromKM:
    """Tests for batch sync from KM."""

    async def test_sync_validations(self):
        """Should sync validations from KM items."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Add some flips
        for i in range(3):
            flip = MockFlipEvent(
                id=f"sync-flip-{i}",
                agent_name="sync-agent",
                original_claim=f"A{i}",
                new_claim=f"B{i}",
                original_confidence=0.7,
                new_confidence=0.8,
                original_debate_id="d1",
                new_debate_id="d2",
                similarity_score=0.8,
            )
            adapter.store_flip(flip)

        km_items = [
            {
                "metadata": {
                    "flip_id": f"fl_sync-flip-{i}",
                    "agent_name": "sync-agent",
                    "similarity_score": 0.8,
                }
            }
            for i in range(3)
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert result.flips_analyzed == 3
        assert result.duration_ms > 0


class TestGetReverseFlowStats:
    """Tests for getting reverse flow statistics."""

    def test_get_reverse_flow_stats_initial(self):
        """Should return initial reverse flow stats."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        stats = adapter.get_reverse_flow_stats()

        assert stats["km_validations_applied"] == 0
        assert stats["km_threshold_updates"] == 0
        assert stats["current_similarity_threshold"] == 0.7


class TestClearReverseFlowState:
    """Tests for clearing reverse flow state."""

    def test_clear_reverse_flow_state(self):
        """Should clear all reverse flow state."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Record some state
        adapter.record_outcome("flip-1", "d1", True)

        # Clear
        adapter.clear_reverse_flow_state()

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_size"] == 0
        assert stats["current_similarity_threshold"] == 0.7


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_store_insight_with_string_type(self):
        """Should handle insight with string type value."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        # Create a simple object that doesn't have a .value attribute
        class StringType:
            pass

        insight = MockInsight(
            id="string-type",
            type=StringType(),  # Not an enum
            title="Test",
            description="Test",
            confidence=0.9,
            debate_id="d1",
        )

        result = adapter.store_insight(insight)

        assert result is not None
        # Type should be converted to string
        assert adapter._insights[result]["type"] is not None

    def test_to_knowledge_item_with_string_created_at(self):
        """Should handle string created_at field."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight_dict = {
            "id": "string-date",
            "title": "Test",
            "description": "Test",
            "confidence": 0.8,
            "created_at": "2024-01-15T10:30:00Z",
        }

        result = adapter.to_knowledge_item(insight_dict)

        assert result is not None
        assert result.created_at is not None

    def test_to_knowledge_item_with_invalid_date(self):
        """Should handle invalid date gracefully."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight_dict = {
            "id": "invalid-date",
            "title": "Test",
            "description": "Test",
            "confidence": 0.8,
            "created_at": "not-a-valid-date",
        }

        result = adapter.to_knowledge_item(insight_dict)

        assert result is not None
        # Should use current time as fallback

    def test_to_knowledge_item_with_none_created_at(self):
        """Should handle None created_at field."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        insight_dict = {
            "id": "none-date",
            "title": "Test",
            "description": "Test",
            "confidence": 0.8,
            "created_at": None,
        }

        result = adapter.to_knowledge_item(insight_dict)

        assert result is not None
        assert result.created_at is not None

    def test_flip_to_knowledge_item_with_missing_fields(self):
        """Should handle flip with missing optional fields."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()
        flip_dict = {
            "id": "minimal-flip",
            # Missing most fields
        }

        result = adapter.flip_to_knowledge_item(flip_dict)

        assert result is not None
        assert result.id == "minimal-flip"

    def test_multiple_insights_same_debate(self):
        """Should index multiple insights for same debate."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(5):
            insight = MockInsight(
                id=f"multi-{i}",
                type=MockInsightType("consensus" if i % 2 == 0 else "pattern"),
                title=f"Insight {i}",
                description="Test",
                confidence=0.9,
                debate_id="shared-debate",
            )
            adapter.store_insight(insight)

        debate_insights = adapter.get_debate_insights("shared-debate")
        assert len(debate_insights) == 5

    def test_multiple_flips_same_agent(self):
        """Should handle multiple flips for same agent."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightsAdapter

        adapter = InsightsAdapter()

        for i in range(10):
            flip = MockFlipEvent(
                id=f"multi-agent-flip-{i}",
                agent_name="prolific-flipper",
                original_claim=f"Position {i}",
                new_claim=f"New position {i}",
                original_confidence=0.6 + (i * 0.02),
                new_confidence=0.7 + (i * 0.02),
                original_debate_id=f"d{i}",
                new_debate_id=f"d{i + 1}",
            )
            adapter.store_flip(flip)

        history = adapter.get_agent_flip_history("prolific-flipper")
        assert len(history) == 10


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestDataclasses:
    """Tests for adapter dataclasses."""

    def test_insight_search_result_defaults(self):
        """Should have correct defaults for InsightSearchResult."""
        from aragora.knowledge.mound.adapters.insights_adapter import InsightSearchResult

        result = InsightSearchResult(insight={"id": "test"})

        assert result.relevance_score == 0.0
        assert result.matched_topics == []

    def test_flip_search_result_defaults(self):
        """Should have correct defaults for FlipSearchResult."""
        from aragora.knowledge.mound.adapters.insights_adapter import FlipSearchResult

        result = FlipSearchResult(flip={"id": "test"})

        assert result.relevance_score == 0.0

    def test_km_flip_threshold_update_defaults(self):
        """Should have correct defaults for KMFlipThresholdUpdate."""
        from aragora.knowledge.mound.adapters.insights_adapter import KMFlipThresholdUpdate

        update = KMFlipThresholdUpdate(
            old_similarity_threshold=0.7,
            new_similarity_threshold=0.75,
            old_confidence_threshold=0.6,
            new_confidence_threshold=0.65,
        )

        assert update.patterns_analyzed == 0
        assert update.adjustments_made == 0
        assert update.confidence == 0.7
        assert update.recommendation == "keep"

    def test_km_agent_flip_baseline_defaults(self):
        """Should have correct defaults for KMAgentFlipBaseline."""
        from aragora.knowledge.mound.adapters.insights_adapter import KMAgentFlipBaseline

        baseline = KMAgentFlipBaseline(
            agent_name="test-agent",
            expected_flip_rate=0.1,
        )

        assert baseline.flip_type_distribution == {}
        assert baseline.domain_flip_rates == {}
        assert baseline.sample_count == 0
        assert baseline.confidence == 0.7

    def test_km_flip_validation_defaults(self):
        """Should have correct defaults for KMFlipValidation."""
        from aragora.knowledge.mound.adapters.insights_adapter import KMFlipValidation

        validation = KMFlipValidation(
            flip_id="test-flip",
            km_confidence=0.8,
        )

        assert validation.is_expected is False
        assert validation.pattern_match_score == 0.0
        assert validation.recommendation == "keep"
        assert validation.adjustment == 0.0

    def test_insight_threshold_sync_result_defaults(self):
        """Should have correct defaults for InsightThresholdSyncResult."""
        from aragora.knowledge.mound.adapters.insights_adapter import (
            InsightThresholdSyncResult,
        )

        result = InsightThresholdSyncResult()

        assert result.flips_analyzed == 0
        assert result.insights_analyzed == 0
        assert result.threshold_updates == []
        assert result.baseline_updates == []
        assert result.errors == []
        assert result.duration_ms == 0.0
