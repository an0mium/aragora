"""Tests for the PulseAdapter."""

import pytest
from unittest.mock import Mock
from datetime import datetime
import time

from aragora.knowledge.mound.adapters.pulse_adapter import (
    PulseAdapter,
    TopicSearchResult,
)
from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource


class TestTopicSearchResult:
    """Tests for TopicSearchResult dataclass."""

    def test_basic_creation(self):
        """Create a basic search result."""
        result = TopicSearchResult(
            topic={"id": "pl_1", "topic": "AI Safety"},
            relevance_score=0.8,
        )
        assert result.topic["id"] == "pl_1"
        assert result.relevance_score == 0.8


class TestPulseAdapterInit:
    """Tests for PulseAdapter initialization."""

    def test_init_without_store(self):
        """Initialize without store."""
        adapter = PulseAdapter()
        assert adapter.debate_store is None

    def test_init_with_store(self):
        """Initialize with store."""
        mock_store = Mock()
        adapter = PulseAdapter(debate_store=mock_store)
        assert adapter.debate_store is mock_store

    def test_constants(self):
        """Verify adapter constants."""
        assert PulseAdapter.ID_PREFIX == "pl_"
        assert PulseAdapter.MIN_TOPIC_QUALITY == 0.6


class TestPulseAdapterQualityScore:
    """Tests for _calculate_quality_score method."""

    def test_high_volume_tech(self):
        """High volume tech topic gets good score."""
        adapter = PulseAdapter()
        score = adapter._calculate_quality_score(volume=100000, category="tech")
        assert score > 0.7

    def test_low_volume_entertainment(self):
        """Low volume entertainment topic gets low score."""
        adapter = PulseAdapter()
        score = adapter._calculate_quality_score(volume=100, category="entertainment")
        assert score < 0.5


class TestPulseAdapterStoreTrendingTopic:
    """Tests for store_trending_topic method."""

    def test_store_quality_topic(self):
        """Store a quality trending topic."""
        adapter = PulseAdapter()

        mock_topic = Mock()
        mock_topic.platform = "twitter"
        mock_topic.topic = "AI Regulation Debate"
        mock_topic.volume = 50000
        mock_topic.category = "tech"
        mock_topic.raw_data = {"tweet_count": 50000}

        topic_id = adapter.store_trending_topic(mock_topic)

        assert topic_id is not None
        assert topic_id.startswith("pl_topic_")

    def test_skip_low_quality_topic(self):
        """Don't store low-quality topics."""
        adapter = PulseAdapter()

        mock_topic = Mock()
        mock_topic.platform = "twitter"
        mock_topic.topic = "Random topic"
        mock_topic.volume = 10
        mock_topic.category = "entertainment"
        mock_topic.raw_data = {}

        topic_id = adapter.store_trending_topic(mock_topic)
        assert topic_id is None

    def test_updates_indices(self):
        """Verify indices are updated."""
        adapter = PulseAdapter()

        mock_topic = Mock()
        mock_topic.platform = "reddit"
        mock_topic.topic = "Machine Learning Advances"
        mock_topic.volume = 100000
        mock_topic.category = "science"
        mock_topic.raw_data = {}

        adapter.store_trending_topic(mock_topic)

        assert "reddit" in adapter._platform_topics
        assert "science" in adapter._category_topics


class TestPulseAdapterStoreScheduledDebate:
    """Tests for store_scheduled_debate method."""

    def test_store_scheduled_debate(self):
        """Store a scheduled debate record."""
        adapter = PulseAdapter()

        mock_record = Mock()
        mock_record.id = "sched_123"
        mock_record.topic_hash = "abc123"
        mock_record.topic_text = "AI Safety Discussion"
        mock_record.platform = "twitter"
        mock_record.category = "tech"
        mock_record.volume = 10000
        mock_record.debate_id = "debate_456"
        mock_record.created_at = time.time()
        mock_record.consensus_reached = True
        mock_record.confidence = 0.85
        mock_record.rounds_used = 5
        mock_record.scheduler_run_id = "run_789"

        debate_id = adapter.store_scheduled_debate(mock_record)

        assert debate_id is not None
        assert debate_id.startswith("pl_debate_")


class TestPulseAdapterStoreOutcome:
    """Tests for store_outcome method."""

    def test_store_outcome(self):
        """Store a debate outcome."""
        adapter = PulseAdapter()

        mock_outcome = Mock()
        mock_outcome.topic = "Climate Policy"
        mock_outcome.platform = "reddit"
        mock_outcome.debate_id = "debate_123"
        mock_outcome.consensus_reached = True
        mock_outcome.confidence = 0.9
        mock_outcome.rounds_used = 4
        mock_outcome.timestamp = time.time()
        mock_outcome.category = "politics"
        mock_outcome.volume = 5000

        outcome_id = adapter.store_outcome(mock_outcome)

        assert outcome_id is not None
        assert outcome_id.startswith("pl_outcome_")


class TestPulseAdapterSearchPastDebates:
    """Tests for search_past_debates method."""

    def test_find_exact_match(self):
        """Find debate with exact topic hash."""
        adapter = PulseAdapter()

        import hashlib

        topic_hash = hashlib.sha256("ai safety".encode()).hexdigest()[:16]

        adapter._debates["pl_d1"] = {
            "id": "pl_d1",
            "topic_hash": topic_hash,
            "topic_text": "AI Safety",
            "created_at": time.time() - 3600,  # 1 hour ago
        }

        results = adapter.search_past_debates("AI Safety", hours=24)

        assert len(results) >= 1
        assert results[0]["match_type"] == "exact"

    def test_find_similar_topics(self):
        """Find debates with similar topics."""
        adapter = PulseAdapter()

        adapter._debates["pl_d1"] = {
            "id": "pl_d1",
            "topic_hash": "different_hash",
            "topic_text": "Machine learning safety concerns",
            "created_at": time.time() - 3600,
        }

        results = adapter.search_past_debates("AI safety machine learning", hours=24)

        # Should find via keyword overlap
        assert len(results) >= 1

    def test_respects_time_window(self):
        """Only find debates within time window."""
        adapter = PulseAdapter()

        adapter._debates["pl_d1"] = {
            "id": "pl_d1",
            "topic_hash": "hash1",
            "topic_text": "Test topic",
            "created_at": time.time() - (72 * 3600),  # 72 hours ago
        }

        results = adapter.search_past_debates("Test topic", hours=24)

        # Should not find (outside window)
        assert len(results) == 0


class TestPulseAdapterGetPlatformTopics:
    """Tests for get_platform_topics method."""

    def test_get_topics_by_platform(self):
        """Get topics from specific platform."""
        adapter = PulseAdapter()

        adapter._topics["pl_t1"] = {"id": "pl_t1", "platform": "twitter"}
        adapter._topics["pl_t2"] = {"id": "pl_t2", "platform": "reddit"}
        adapter._platform_topics["twitter"] = ["pl_t1"]
        adapter._platform_topics["reddit"] = ["pl_t2"]

        results = adapter.get_platform_topics("twitter")

        assert len(results) == 1
        assert results[0]["platform"] == "twitter"


class TestPulseAdapterGetTrendingPatterns:
    """Tests for get_trending_patterns method."""

    def test_find_patterns(self):
        """Find recurring keyword patterns."""
        adapter = PulseAdapter()

        adapter._topics["t1"] = {"id": "t1", "topic": "AI safety concerns"}
        adapter._topics["t2"] = {"id": "t2", "topic": "AI safety regulation"}
        adapter._topics["t3"] = {"id": "t3", "topic": "AI safety research"}

        patterns = adapter.get_trending_patterns(min_occurrences=3)

        # Should find "safety" as a pattern
        keywords = [p["keyword"] for p in patterns]
        assert "safety" in keywords


class TestPulseAdapterGetOutcomeAnalytics:
    """Tests for get_outcome_analytics method."""

    def test_get_analytics(self):
        """Get outcome analytics."""
        adapter = PulseAdapter()

        adapter._outcomes["o1"] = {
            "platform": "twitter",
            "category": "tech",
            "consensus_reached": True,
            "confidence": 0.8,
            "rounds_used": 4,
        }
        adapter._outcomes["o2"] = {
            "platform": "twitter",
            "category": "tech",
            "consensus_reached": False,
            "confidence": 0.5,
            "rounds_used": 8,
        }

        analytics = adapter.get_outcome_analytics(platform="twitter")

        assert analytics["total_debates"] == 2
        assert analytics["consensus_rate"] == 0.5
        assert analytics["avg_confidence"] == 0.65

    def test_empty_analytics(self):
        """Handle empty outcome set."""
        adapter = PulseAdapter()

        analytics = adapter.get_outcome_analytics()

        assert analytics["total_debates"] == 0
        assert analytics["consensus_rate"] == 0.0


class TestPulseAdapterToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_convert_topic(self):
        """Convert topic to knowledge item."""
        adapter = PulseAdapter()

        topic = {
            "id": "pl_topic_abc123",
            "topic_hash": "abc123",
            "topic": "AI Regulation Debate",
            "platform": "twitter",
            "category": "tech",
            "volume": 50000,
            "quality_score": 0.85,
            "created_at": "2024-01-01T00:00:00Z",
        }

        item = adapter.to_knowledge_item(topic)

        assert item.id == "pl_topic_abc123"
        assert item.content == "AI Regulation Debate"
        assert item.source == KnowledgeSource.PULSE
        assert item.confidence == ConfidenceLevel.HIGH
        assert item.metadata["platform"] == "twitter"


class TestPulseAdapterGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self):
        """Get adapter statistics."""
        adapter = PulseAdapter()

        adapter._topics["t1"] = {}
        adapter._debates["d1"] = {}
        adapter._outcomes["o1"] = {}
        adapter._platform_topics["twitter"] = ["t1"]
        adapter._category_topics["tech"] = ["t1"]

        stats = adapter.get_stats()

        assert stats["total_topics"] == 1
        assert stats["total_debates"] == 1
        assert stats["total_outcomes"] == 1
        assert stats["platforms_tracked"] == 1
        assert stats["categories_tracked"] == 1
