"""Tests for the InsightsAdapter."""

import pytest
from unittest.mock import Mock
from datetime import datetime

from aragora.knowledge.mound.adapters.insights_adapter import (
    InsightsAdapter,
    InsightSearchResult,
    FlipSearchResult,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


class TestInsightSearchResult:
    """Tests for InsightSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic search result."""
        result = InsightSearchResult(
            insight={"id": "in_1", "title": "test"},
            relevance_score=0.8,
        )
        assert result.insight["id"] == "in_1"
        assert result.relevance_score == 0.8


class TestFlipSearchResult:
    """Tests for FlipSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic flip search result."""
        result = FlipSearchResult(
            flip={"id": "fl_1", "agent_name": "agent1"},
            relevance_score=0.9,
        )
        assert result.flip["id"] == "fl_1"


class TestInsightsAdapterInit:
    """Tests for InsightsAdapter initialization."""

    def test_init_without_stores(self):
        """Initialize without stores."""
        adapter = InsightsAdapter()
        assert adapter.insight_store is None
        assert adapter.flip_detector is None

    def test_constants(self):
        """Verify adapter constants."""
        assert InsightsAdapter.INSIGHT_PREFIX == "in_"
        assert InsightsAdapter.FLIP_PREFIX == "fl_"
        assert InsightsAdapter.PATTERN_PREFIX == "pt_"
        assert InsightsAdapter.MIN_INSIGHT_CONFIDENCE == 0.7
        assert InsightsAdapter.MIN_PATTERN_OCCURRENCES == 3


class TestInsightsAdapterStoreInsight:
    """Tests for store_insight method."""

    def test_store_high_confidence_insight(self):
        """Store a high-confidence insight."""
        adapter = InsightsAdapter()

        mock_insight = Mock()
        mock_insight.id = "ins_123"
        mock_insight.type = Mock(value="consensus")
        mock_insight.title = "Key Finding"
        mock_insight.description = "Important discovery"
        mock_insight.confidence = 0.85
        mock_insight.debate_id = "debate_1"
        mock_insight.agents_involved = ["agent1", "agent2"]
        mock_insight.evidence = ["ev1"]
        mock_insight.created_at = "2024-01-01T00:00:00Z"
        mock_insight.metadata = {}

        insight_id = adapter.store_insight(mock_insight)

        assert insight_id is not None
        assert insight_id.startswith("in_")

    def test_skip_low_confidence_insight(self):
        """Don't store insights below threshold."""
        adapter = InsightsAdapter()

        mock_insight = Mock()
        mock_insight.id = "ins_123"
        mock_insight.confidence = 0.5  # Below 0.7

        insight_id = adapter.store_insight(mock_insight)
        assert insight_id is None


class TestInsightsAdapterStoreFlip:
    """Tests for store_flip method."""

    def test_store_flip_event(self):
        """Store a flip event (always stored)."""
        adapter = InsightsAdapter()

        mock_flip = Mock()
        mock_flip.id = "flip_123"
        mock_flip.agent_name = "agent1"
        mock_flip.original_claim = "Original position"
        mock_flip.new_claim = "New position"
        mock_flip.original_confidence = 0.8
        mock_flip.new_confidence = 0.7
        mock_flip.original_debate_id = "debate_1"
        mock_flip.new_debate_id = "debate_2"
        mock_flip.original_position_id = "pos_1"
        mock_flip.new_position_id = "pos_2"
        mock_flip.similarity_score = 0.85
        mock_flip.flip_type = "contradiction"
        mock_flip.domain = "legal"
        mock_flip.detected_at = "2024-01-01T00:00:00Z"

        flip_id = adapter.store_flip(mock_flip)

        assert flip_id is not None
        assert flip_id.startswith("fl_")
        assert "agent1" in adapter._agent_flips

    def test_store_flip_updates_domain_index(self):
        """Verify domain index is updated."""
        adapter = InsightsAdapter()

        mock_flip = Mock()
        mock_flip.id = "flip_123"
        mock_flip.agent_name = "agent1"
        mock_flip.original_claim = "Test"
        mock_flip.new_claim = "Test2"
        mock_flip.original_confidence = 0.8
        mock_flip.new_confidence = 0.7
        mock_flip.original_debate_id = "d1"
        mock_flip.new_debate_id = "d2"
        mock_flip.original_position_id = "p1"
        mock_flip.new_position_id = "p2"
        mock_flip.similarity_score = 0.8
        mock_flip.flip_type = "refinement"
        mock_flip.domain = "tech"
        mock_flip.detected_at = "2024-01-01T00:00:00Z"

        adapter.store_flip(mock_flip)

        assert "tech" in adapter._domain_flips


class TestInsightsAdapterStorePattern:
    """Tests for store_pattern method."""

    def test_store_pattern_above_threshold(self):
        """Store pattern with sufficient occurrences."""
        adapter = InsightsAdapter()

        pattern_id = adapter.store_pattern(
            category="argument_style",
            pattern_text="Agents often cite evidence before conclusions",
            occurrence_count=5,
            avg_severity=0.6,
            debate_ids=["d1", "d2", "d3"],
        )

        assert pattern_id is not None
        assert pattern_id.startswith("pt_")

    def test_skip_pattern_below_threshold(self):
        """Don't store patterns below occurrence threshold."""
        adapter = InsightsAdapter()

        pattern_id = adapter.store_pattern(
            category="test",
            pattern_text="Test pattern",
            occurrence_count=2,  # Below 3
        )

        assert pattern_id is None

    def test_pattern_aggregation(self):
        """Test that patterns aggregate on re-store."""
        adapter = InsightsAdapter()

        # Store first time
        adapter.store_pattern(
            category="style",
            pattern_text="Test pattern",
            occurrence_count=3,
            debate_ids=["d1"],
        )

        # Store again (should aggregate - need 3+ to store)
        adapter.store_pattern(
            category="style",
            pattern_text="Test pattern",
            occurrence_count=3,
            debate_ids=["d2"],
        )

        # Check aggregation
        assert len(adapter._patterns) == 1
        pattern = list(adapter._patterns.values())[0]
        assert pattern["occurrence_count"] == 6


class TestInsightsAdapterGetAgentFlipHistory:
    """Tests for get_agent_flip_history method."""

    def test_get_flip_history(self):
        """Get flip history for an agent."""
        adapter = InsightsAdapter()

        adapter._flips["fl_1"] = {
            "id": "fl_1",
            "agent_name": "agent1",
            "flip_type": "contradiction",
            "detected_at": "2024-01-02T00:00:00Z",
        }
        adapter._flips["fl_2"] = {
            "id": "fl_2",
            "agent_name": "agent1",
            "flip_type": "refinement",
            "detected_at": "2024-01-01T00:00:00Z",
        }
        adapter._agent_flips["agent1"] = ["fl_1", "fl_2"]

        results = adapter.get_agent_flip_history("agent1")

        assert len(results) == 2
        # Should be sorted newest first
        assert results[0]["detected_at"] > results[1]["detected_at"]

    def test_filter_by_flip_type(self):
        """Filter flip history by type."""
        adapter = InsightsAdapter()

        adapter._flips["fl_1"] = {
            "id": "fl_1",
            "flip_type": "contradiction",
            "detected_at": "2024-01-01T00:00:00Z",
        }
        adapter._flips["fl_2"] = {
            "id": "fl_2",
            "flip_type": "refinement",
            "detected_at": "2024-01-01T00:00:00Z",
        }
        adapter._agent_flips["agent1"] = ["fl_1", "fl_2"]

        results = adapter.get_agent_flip_history("agent1", flip_type="contradiction")

        assert len(results) == 1
        assert results[0]["flip_type"] == "contradiction"


class TestInsightsAdapterSearchSimilarInsights:
    """Tests for search_similar_insights method."""

    def test_search_finds_matches(self):
        """Find matching insights."""
        adapter = InsightsAdapter()

        adapter._insights["in_1"] = {
            "id": "in_1",
            "title": "Consensus on data privacy",
            "description": "Agents agreed on privacy controls",
            "type": "consensus",
            "confidence": 0.9,
        }

        results = adapter.search_similar_insights("data privacy")

        assert len(results) >= 1

    def test_filter_by_type(self):
        """Filter insights by type."""
        adapter = InsightsAdapter()

        adapter._insights["in_1"] = {
            "id": "in_1",
            "title": "Test insight",
            "description": "Description",
            "type": "consensus",
            "confidence": 0.9,
        }
        adapter._type_insights["consensus"] = ["in_1"]

        results = adapter.search_similar_insights("test", insight_type="pattern")
        assert len(results) == 0


class TestInsightsAdapterToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_convert_insight(self):
        """Convert insight to knowledge item."""
        adapter = InsightsAdapter()

        insight = {
            "id": "in_123",
            "original_id": "ins_123",
            "title": "Key Finding",
            "description": "Important discovery about debate dynamics",
            "type": "consensus",
            "confidence": 0.9,
            "debate_id": "debate_1",
            "agents_involved": ["agent1", "agent2"],
            "evidence": ["ev1"],
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(insight)

        assert item.id == "in_123"
        assert "Key Finding" in item.content
        assert item.source == KnowledgeSource.INSIGHT
        assert item.confidence == ConfidenceLevel.VERIFIED


class TestInsightsAdapterFlipToKnowledgeItem:
    """Tests for flip_to_knowledge_item method."""

    def test_convert_flip(self):
        """Convert flip to knowledge item."""
        adapter = InsightsAdapter()

        flip = {
            "id": "fl_123",
            "original_id": "flip_123",
            "agent_name": "agent1",
            "original_claim": "Original position on topic",
            "new_claim": "New position on topic",
            "flip_type": "contradiction",
            "domain": "legal",
            "similarity_score": 0.85,
            "detected_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.flip_to_knowledge_item(flip)

        assert item.id == "fl_123"
        assert "agent1" in item.content
        assert item.source == KnowledgeSource.FLIP
        assert item.metadata["flip_type"] == "contradiction"


class TestInsightsAdapterGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self):
        """Get adapter statistics."""
        adapter = InsightsAdapter()

        adapter._insights["in_1"] = {"type": "consensus"}
        adapter._flips["fl_1"] = {"flip_type": "contradiction"}
        adapter._flips["fl_2"] = {"flip_type": "refinement"}
        adapter._patterns["pt_1"] = {}
        adapter._type_insights["consensus"] = ["in_1"]
        adapter._agent_flips["agent1"] = ["fl_1", "fl_2"]

        stats = adapter.get_stats()

        assert stats["total_insights"] == 1
        assert stats["total_flips"] == 2
        assert stats["total_patterns"] == 1
        assert stats["flip_types"]["contradiction"] == 1
        assert stats["flip_types"]["refinement"] == 1
