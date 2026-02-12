"""
Tests for PulseAdapter - Bridges Pulse (trending topics) to Knowledge Mound.

Tests cover:
- Adapter initialization and configuration
- Topic ingestion from different sources
- Freshness scoring calculations
- Source weighting logic
- Topic retrieval and filtering
- Quality filtering
- Integration with KnowledgeMound
- Error handling and edge cases
- Reverse flow methods (KM -> Pulse)
"""

import hashlib
import math
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_trending_topic():
    """Create a mock TrendingTopic."""
    topic = MagicMock()
    topic.platform = "hackernews"
    topic.topic = "AI language models revolutionizing code"
    topic.volume = 10000
    topic.category = "tech"
    topic.raw_data = {"url": "https://news.ycombinator.com/item?id=12345"}
    return topic


@pytest.fixture
def mock_scheduled_debate_record():
    """Create a mock ScheduledDebateRecord."""
    record = MagicMock()
    record.id = "sched_001"
    record.topic_hash = "abc123"
    record.topic_text = "Debate on AI safety"
    record.platform = "hackernews"
    record.category = "tech"
    record.volume = 5000
    record.debate_id = "debate_001"
    record.created_at = time.time()
    record.consensus_reached = True
    record.confidence = 0.85
    record.rounds_used = 3
    record.scheduler_run_id = "run_001"
    return record


@pytest.fixture
def mock_trending_topic_outcome():
    """Create a mock TrendingTopicOutcome."""
    outcome = MagicMock()
    outcome.topic = "AI language models discussion"
    outcome.platform = "hackernews"
    outcome.debate_id = "debate_002"
    outcome.consensus_reached = True
    outcome.confidence = 0.9
    outcome.rounds_used = 4
    outcome.timestamp = time.time()
    outcome.category = "tech"
    outcome.volume = 8000
    return outcome


@pytest.fixture
def adapter():
    """Create a PulseAdapter instance."""
    from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

    return PulseAdapter()


@pytest.fixture
def adapter_with_store():
    """Create a PulseAdapter with a mock debate store."""
    from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

    mock_store = MagicMock()
    return PulseAdapter(debate_store=mock_store)


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestKMQualityThresholdUpdate:
    """Tests for KMQualityThresholdUpdate dataclass."""

    def test_create_threshold_update(self):
        """Should create KMQualityThresholdUpdate with all fields."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMQualityThresholdUpdate

        update = KMQualityThresholdUpdate(
            old_min_quality=0.6,
            new_min_quality=0.5,
            old_category_bonuses={"tech": 0.2},
            new_category_bonuses={"tech": 0.25},
            patterns_analyzed=100,
            adjustments_made=3,
            confidence=0.8,
            recommendation="lower_threshold",
            metadata={"source": "test"},
        )

        assert update.old_min_quality == 0.6
        assert update.new_min_quality == 0.5
        assert update.patterns_analyzed == 100
        assert update.adjustments_made == 3
        assert update.recommendation == "lower_threshold"

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMQualityThresholdUpdate

        update = KMQualityThresholdUpdate(
            old_min_quality=0.6,
            new_min_quality=0.6,
        )

        assert update.old_category_bonuses == {}
        assert update.new_category_bonuses == {}
        assert update.patterns_analyzed == 0
        assert update.adjustments_made == 0
        assert update.confidence == 0.7
        assert update.recommendation == "keep"


class TestKMTopicCoverage:
    """Tests for KMTopicCoverage dataclass."""

    def test_create_coverage(self):
        """Should create KMTopicCoverage with all fields."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicCoverage

        coverage = KMTopicCoverage(
            topic_text="AI safety discussion",
            coverage_score=0.75,
            related_debates_count=5,
            avg_outcome_confidence=0.82,
            consensus_rate=0.8,
            km_items_found=10,
            recommendation="proceed",
            priority_adjustment=0.1,
            metadata={"topic_hash": "abc123"},
        )

        assert coverage.topic_text == "AI safety discussion"
        assert coverage.coverage_score == 0.75
        assert coverage.related_debates_count == 5
        assert coverage.recommendation == "proceed"

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicCoverage

        coverage = KMTopicCoverage(topic_text="Test topic")

        assert coverage.coverage_score == 0.0
        assert coverage.related_debates_count == 0
        assert coverage.avg_outcome_confidence == 0.0
        assert coverage.consensus_rate == 0.0
        assert coverage.km_items_found == 0
        assert coverage.recommendation == "proceed"
        assert coverage.priority_adjustment == 0.0


class TestKMSchedulingRecommendation:
    """Tests for KMSchedulingRecommendation dataclass."""

    def test_create_recommendation(self):
        """Should create KMSchedulingRecommendation with all fields."""
        from aragora.knowledge.mound.adapters.pulse_adapter import (
            KMSchedulingRecommendation,
            KMTopicCoverage,
        )

        coverage = KMTopicCoverage(topic_text="Test topic")
        rec = KMSchedulingRecommendation(
            topic_id="pl_topic_001",
            original_priority=0.5,
            adjusted_priority=0.6,
            reason="boost",
            km_confidence=0.85,
            coverage=coverage,
            was_applied=True,
            metadata={"adjustment": 0.1},
        )

        assert rec.topic_id == "pl_topic_001"
        assert rec.original_priority == 0.5
        assert rec.adjusted_priority == 0.6
        assert rec.was_applied is True

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMSchedulingRecommendation

        rec = KMSchedulingRecommendation(topic_id="test_001")

        assert rec.original_priority == 0.5
        assert rec.adjusted_priority == 0.5
        assert rec.reason == "no_change"
        assert rec.km_confidence == 0.7
        assert rec.coverage is None
        assert rec.was_applied is False


class TestKMTopicValidation:
    """Tests for KMTopicValidation dataclass."""

    def test_create_validation(self):
        """Should create KMTopicValidation with all fields."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicValidation

        validation = KMTopicValidation(
            topic_id="pl_topic_001",
            km_confidence=0.9,
            outcome_success_rate=0.85,
            similar_debates_count=10,
            avg_rounds_needed=3.5,
            recommendation="boost",
            priority_adjustment=0.1,
            metadata={"source": "test"},
        )

        assert validation.topic_id == "pl_topic_001"
        assert validation.km_confidence == 0.9
        assert validation.outcome_success_rate == 0.85
        assert validation.recommendation == "boost"

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicValidation

        validation = KMTopicValidation(topic_id="test_001")

        assert validation.km_confidence == 0.7
        assert validation.outcome_success_rate == 0.0
        assert validation.similar_debates_count == 0
        assert validation.avg_rounds_needed == 0.0
        assert validation.recommendation == "keep"
        assert validation.priority_adjustment == 0.0


class TestPulseKMSyncResult:
    """Tests for PulseKMSyncResult dataclass."""

    def test_create_sync_result(self):
        """Should create PulseKMSyncResult with all fields."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseKMSyncResult

        result = PulseKMSyncResult(
            topics_analyzed=50,
            topics_adjusted=10,
            threshold_updates=2,
            scheduling_changes=5,
            errors=["error1", "error2"],
            duration_ms=150,
            metadata={"source": "test"},
        )

        assert result.topics_analyzed == 50
        assert result.topics_adjusted == 10
        assert result.threshold_updates == 2
        assert len(result.errors) == 2

    def test_default_values(self):
        """Should have sensible defaults."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseKMSyncResult

        result = PulseKMSyncResult()

        assert result.topics_analyzed == 0
        assert result.topics_adjusted == 0
        assert result.threshold_updates == 0
        assert result.scheduling_changes == 0
        assert result.errors == []
        assert result.duration_ms == 0


class TestTopicSearchResult:
    """Tests for TopicSearchResult dataclass."""

    def test_create_search_result(self):
        """Should create TopicSearchResult with topic and relevance."""
        from aragora.knowledge.mound.adapters.pulse_adapter import TopicSearchResult

        topic_data = {"id": "topic_001", "topic": "AI safety"}
        result = TopicSearchResult(topic=topic_data, relevance_score=0.85)

        assert result.topic["id"] == "topic_001"
        assert result.relevance_score == 0.85

    def test_default_relevance(self):
        """Should default relevance_score to 0.0."""
        from aragora.knowledge.mound.adapters.pulse_adapter import TopicSearchResult

        result = TopicSearchResult(topic={})

        assert result.relevance_score == 0.0


# =============================================================================
# Adapter Initialization Tests
# =============================================================================


class TestPulseAdapterInit:
    """Tests for PulseAdapter initialization."""

    def test_init_default(self):
        """Should initialize with default values."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        adapter = PulseAdapter()

        assert adapter._debate_store is None
        assert adapter._enable_dual_write is False
        assert adapter._event_callback is None
        assert adapter._topics == {}
        assert adapter._debates == {}
        assert adapter._outcomes == {}

    def test_init_with_debate_store(self):
        """Should initialize with debate store."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        mock_store = MagicMock()
        adapter = PulseAdapter(debate_store=mock_store)

        assert adapter._debate_store is mock_store

    def test_init_with_dual_write(self):
        """Should initialize with dual write enabled."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        adapter = PulseAdapter(enable_dual_write=True)

        assert adapter._enable_dual_write is True

    def test_init_with_event_callback(self):
        """Should initialize with event callback."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        callback = MagicMock()
        adapter = PulseAdapter(event_callback=callback)

        assert adapter._event_callback is callback

    def test_debate_store_property(self):
        """Should access debate store via property."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        mock_store = MagicMock()
        adapter = PulseAdapter(debate_store=mock_store)

        assert adapter.debate_store is mock_store

    def test_id_prefix(self):
        """Should have correct ID prefix."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        assert PulseAdapter.ID_PREFIX == "pl_"

    def test_min_topic_quality_threshold(self):
        """Should have correct minimum topic quality threshold."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        assert PulseAdapter.MIN_TOPIC_QUALITY == 0.6


# =============================================================================
# Event Callback Tests
# =============================================================================


class TestEventCallback:
    """Tests for event callback functionality."""

    def test_set_event_callback(self, adapter):
        """Should set event callback."""
        callback = MagicMock()
        adapter.set_event_callback(callback)

        assert adapter._event_callback is callback

    def test_emit_event(self, adapter):
        """Should emit event via callback."""
        callback = MagicMock()
        adapter.set_event_callback(callback)

        adapter._emit_event("topic_stored", {"topic_id": "test_001"})

        callback.assert_called_once_with("topic_stored", {"topic_id": "test_001"})

    def test_emit_event_no_callback(self, adapter):
        """Should not fail when no callback is set."""
        # Should not raise
        adapter._emit_event("topic_stored", {"topic_id": "test_001"})

    def test_emit_event_handles_callback_error(self, adapter):
        """Should handle callback errors gracefully."""
        callback = MagicMock(side_effect=Exception("Callback failed"))
        adapter.set_event_callback(callback)

        # Should not raise
        adapter._emit_event("topic_stored", {"topic_id": "test_001"})


# =============================================================================
# Quality Score Calculation Tests
# =============================================================================


class TestQualityScoreCalculation:
    """Tests for _calculate_quality_score method."""

    def test_quality_score_low_volume(self, adapter):
        """Should calculate low quality score for low volume."""
        score = adapter._calculate_quality_score(volume=10, category="tech")

        # log10(10) = 1, score = 1/6 + 0.2 (tech bonus) = ~0.367
        assert 0.3 <= score <= 0.4

    def test_quality_score_medium_volume(self, adapter):
        """Should calculate medium quality score for medium volume."""
        score = adapter._calculate_quality_score(volume=1000, category="tech")

        # log10(1000) = 3, score = 3/6 + 0.2 = 0.7
        assert score == pytest.approx(0.7, abs=0.01)

    def test_quality_score_high_volume(self, adapter):
        """Should calculate high quality score for high volume."""
        score = adapter._calculate_quality_score(volume=1000000, category="tech")

        # log10(1000000) = 6, score = 6/6 + 0.2 = 1.2 -> capped at 1.0
        assert score == 1.0

    def test_quality_score_tech_bonus(self, adapter):
        """Should apply tech category bonus."""
        tech_score = adapter._calculate_quality_score(volume=1000, category="tech")
        base_score = adapter._calculate_quality_score(volume=1000, category="other")

        assert tech_score > base_score
        assert tech_score - base_score == pytest.approx(0.2, abs=0.01)

    def test_quality_score_science_bonus(self, adapter):
        """Should apply science category bonus."""
        science_score = adapter._calculate_quality_score(volume=1000, category="science")
        base_score = adapter._calculate_quality_score(volume=1000, category="other")

        assert science_score > base_score
        assert science_score - base_score == pytest.approx(0.2, abs=0.01)

    def test_quality_score_business_bonus(self, adapter):
        """Should apply business category bonus."""
        business_score = adapter._calculate_quality_score(volume=1000, category="business")
        base_score = adapter._calculate_quality_score(volume=1000, category="other")

        assert business_score > base_score
        assert business_score - base_score == pytest.approx(0.1, abs=0.01)

    def test_quality_score_politics_no_bonus(self, adapter):
        """Should not apply bonus for politics category."""
        politics_score = adapter._calculate_quality_score(volume=1000, category="politics")
        base_score = adapter._calculate_quality_score(volume=1000, category="other")

        assert politics_score == pytest.approx(base_score, abs=0.01)

    def test_quality_score_entertainment_penalty(self, adapter):
        """Should apply entertainment category penalty."""
        entertainment_score = adapter._calculate_quality_score(
            volume=1000, category="entertainment"
        )
        base_score = adapter._calculate_quality_score(volume=1000, category="other")

        assert entertainment_score < base_score
        assert base_score - entertainment_score == pytest.approx(0.1, abs=0.01)

    def test_quality_score_zero_volume(self, adapter):
        """Should handle zero volume."""
        score = adapter._calculate_quality_score(volume=0, category="tech")

        # log10(1) = 0, score = 0/6 + 0.2 = 0.2
        assert score == pytest.approx(0.2, abs=0.01)

    def test_quality_score_negative_volume(self, adapter):
        """Should handle negative volume gracefully."""
        score = adapter._calculate_quality_score(volume=-100, category="tech")

        # max(1, -100) = 1, log10(1) = 0, score = 0 + 0.2 = 0.2
        assert score == pytest.approx(0.2, abs=0.01)

    def test_quality_score_capped_at_one(self, adapter):
        """Should cap quality score at 1.0."""
        score = adapter._calculate_quality_score(volume=10000000, category="tech")

        assert score == 1.0

    def test_quality_score_capped_at_zero(self, adapter):
        """Should cap quality score at 0.0."""
        # Very low volume with entertainment penalty
        score = adapter._calculate_quality_score(volume=1, category="entertainment")

        # Should be >= 0
        assert score >= 0.0

    def test_quality_score_case_insensitive_category(self, adapter):
        """Should handle category case insensitively."""
        score_upper = adapter._calculate_quality_score(volume=1000, category="TECH")
        score_lower = adapter._calculate_quality_score(volume=1000, category="tech")
        score_mixed = adapter._calculate_quality_score(volume=1000, category="Tech")

        assert score_upper == score_lower == score_mixed


# =============================================================================
# Store Trending Topic Tests
# =============================================================================


class TestStoreTrendingTopic:
    """Tests for store_trending_topic method."""

    def test_store_high_quality_topic(self, adapter, mock_trending_topic):
        """Should store topic with quality above threshold."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        assert topic_id is not None
        assert topic_id.startswith("pl_topic_")
        assert topic_id in adapter._topics

    def test_store_topic_generates_hash(self, adapter, mock_trending_topic):
        """Should generate topic hash from topic text."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        stored_topic = adapter._topics[topic_id]
        expected_hash = hashlib.sha256(mock_trending_topic.topic.lower().encode()).hexdigest()[:16]
        assert stored_topic["topic_hash"] == expected_hash

    def test_reject_low_quality_topic(self, adapter, mock_trending_topic):
        """Should reject topic below quality threshold."""
        mock_trending_topic.volume = 1  # Very low volume
        mock_trending_topic.category = "entertainment"  # Penalty category

        topic_id = adapter.store_trending_topic(mock_trending_topic)

        assert topic_id is None
        assert len(adapter._topics) == 0

    def test_custom_min_quality(self, adapter, mock_trending_topic):
        """Should respect custom min_quality threshold."""
        mock_trending_topic.volume = 100  # Moderate volume

        # Should pass with low threshold
        topic_id = adapter.store_trending_topic(mock_trending_topic, min_quality=0.3)
        assert topic_id is not None

    def test_store_topic_populates_data(self, adapter, mock_trending_topic):
        """Should populate topic data correctly."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        stored = adapter._topics[topic_id]
        assert stored["topic"] == mock_trending_topic.topic
        assert stored["platform"] == mock_trending_topic.platform
        assert stored["volume"] == mock_trending_topic.volume
        assert stored["category"] == mock_trending_topic.category
        assert stored["raw_data"] == mock_trending_topic.raw_data
        assert "created_at" in stored
        assert "quality_score" in stored

    def test_store_topic_updates_platform_index(self, adapter, mock_trending_topic):
        """Should update platform index."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        assert mock_trending_topic.platform in adapter._platform_topics
        assert topic_id in adapter._platform_topics[mock_trending_topic.platform]

    def test_store_topic_updates_category_index(self, adapter, mock_trending_topic):
        """Should update category index."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        assert mock_trending_topic.category in adapter._category_topics
        assert topic_id in adapter._category_topics[mock_trending_topic.category]

    def test_store_topic_updates_hash_map(self, adapter, mock_trending_topic):
        """Should update topic hash map."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        expected_hash = hashlib.sha256(mock_trending_topic.topic.lower().encode()).hexdigest()[:16]
        assert expected_hash in adapter._topic_hash_map
        assert adapter._topic_hash_map[expected_hash] == topic_id

    def test_store_topic_empty_category(self, adapter, mock_trending_topic):
        """Should handle empty category."""
        mock_trending_topic.category = ""

        topic_id = adapter.store_trending_topic(mock_trending_topic)

        assert topic_id is not None
        # Empty category should not be added to index
        assert "" not in adapter._category_topics or topic_id not in adapter._category_topics.get(
            "", []
        )

    def test_store_multiple_topics_same_platform(self, adapter):
        """Should store multiple topics from same platform."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        topic1 = MagicMock()
        topic1.platform = "hackernews"
        topic1.topic = "Topic one"
        topic1.volume = 10000
        topic1.category = "tech"
        topic1.raw_data = {}

        topic2 = MagicMock()
        topic2.platform = "hackernews"
        topic2.topic = "Topic two"
        topic2.volume = 20000
        topic2.category = "tech"
        topic2.raw_data = {}

        id1 = adapter.store_trending_topic(topic1)
        id2 = adapter.store_trending_topic(topic2)

        assert len(adapter._platform_topics["hackernews"]) == 2
        assert id1 in adapter._platform_topics["hackernews"]
        assert id2 in adapter._platform_topics["hackernews"]


# =============================================================================
# Store Scheduled Debate Tests
# =============================================================================


class TestStoreScheduledDebate:
    """Tests for store_scheduled_debate method."""

    def test_store_scheduled_debate(self, adapter, mock_scheduled_debate_record):
        """Should store scheduled debate record."""
        debate_id = adapter.store_scheduled_debate(mock_scheduled_debate_record)

        assert debate_id is not None
        assert debate_id.startswith("pl_debate_")
        assert debate_id in adapter._debates

    def test_store_debate_populates_data(self, adapter, mock_scheduled_debate_record):
        """Should populate debate data correctly."""
        debate_id = adapter.store_scheduled_debate(mock_scheduled_debate_record)

        stored = adapter._debates[debate_id]
        assert stored["original_id"] == mock_scheduled_debate_record.id
        assert stored["topic_hash"] == mock_scheduled_debate_record.topic_hash
        assert stored["topic_text"] == mock_scheduled_debate_record.topic_text
        assert stored["platform"] == mock_scheduled_debate_record.platform
        assert stored["consensus_reached"] == mock_scheduled_debate_record.consensus_reached
        assert "stored_at" in stored

    def test_store_multiple_debates(self, adapter, mock_scheduled_debate_record):
        """Should store multiple debate records."""
        id1 = adapter.store_scheduled_debate(mock_scheduled_debate_record)

        record2 = MagicMock()
        record2.id = "sched_002"
        record2.topic_hash = "def456"
        record2.topic_text = "Another debate"
        record2.platform = "reddit"
        record2.category = "science"
        record2.volume = 3000
        record2.debate_id = "debate_002"
        record2.created_at = time.time()
        record2.consensus_reached = False
        record2.confidence = 0.6
        record2.rounds_used = 5
        record2.scheduler_run_id = "run_002"

        id2 = adapter.store_scheduled_debate(record2)

        assert len(adapter._debates) == 2
        assert id1 != id2


# =============================================================================
# Store Outcome Tests
# =============================================================================


class TestStoreOutcome:
    """Tests for store_outcome method."""

    def test_store_outcome(self, adapter, mock_trending_topic_outcome):
        """Should store debate outcome."""
        outcome_id = adapter.store_outcome(mock_trending_topic_outcome)

        assert outcome_id is not None
        assert outcome_id.startswith("pl_outcome_")
        assert outcome_id in adapter._outcomes

    def test_store_outcome_populates_data(self, adapter, mock_trending_topic_outcome):
        """Should populate outcome data correctly."""
        outcome_id = adapter.store_outcome(mock_trending_topic_outcome)

        stored = adapter._outcomes[outcome_id]
        assert stored["topic"] == mock_trending_topic_outcome.topic
        assert stored["platform"] == mock_trending_topic_outcome.platform
        assert stored["debate_id"] == mock_trending_topic_outcome.debate_id
        assert stored["consensus_reached"] == mock_trending_topic_outcome.consensus_reached
        assert stored["confidence"] == mock_trending_topic_outcome.confidence
        assert "topic_hash" in stored
        assert "stored_at" in stored

    def test_store_outcome_generates_hash(self, adapter, mock_trending_topic_outcome):
        """Should generate topic hash from topic text."""
        outcome_id = adapter.store_outcome(mock_trending_topic_outcome)

        stored = adapter._outcomes[outcome_id]
        expected_hash = hashlib.sha256(
            mock_trending_topic_outcome.topic.lower().encode()
        ).hexdigest()[:16]
        assert stored["topic_hash"] == expected_hash


class TestStoreDebateOutcome:
    """Tests for store_debate_outcome convenience method."""

    def test_store_debate_outcome(self, adapter):
        """Should store debate outcome via convenience method."""
        with patch("aragora.knowledge.mound.adapters.pulse_adapter.time") as mock_time:
            mock_time.time.return_value = 1234567890

            outcome_id = adapter.store_debate_outcome(
                debate_id="debate_123",
                topic="Test topic",
                platform="hackernews",
                consensus_reached=True,
                confidence=0.85,
                rounds_used=3,
                category="tech",
                volume=5000,
            )

        assert outcome_id is not None
        assert outcome_id.startswith("pl_outcome_")

    def test_store_debate_outcome_records_for_km(self, adapter):
        """Should record outcome for KM reverse flow."""
        adapter._init_reverse_flow_state()

        adapter.store_debate_outcome(
            debate_id="debate_123",
            topic="Test topic",
            platform="hackernews",
            consensus_reached=True,
            confidence=0.85,
            rounds_used=3,
        )

        # Should have recorded for KM
        assert len(adapter._outcome_history) == 1


# =============================================================================
# Get Topic/Debate Tests
# =============================================================================


class TestGetTopic:
    """Tests for get_topic method."""

    def test_get_topic_with_prefix(self, adapter, mock_trending_topic):
        """Should get topic with pl_topic_ prefix."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        result = adapter.get_topic(topic_id)

        assert result is not None
        assert result["id"] == topic_id

    def test_get_topic_without_prefix(self, adapter, mock_trending_topic):
        """Should get topic without prefix."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)
        topic_hash = topic_id.replace("pl_topic_", "")

        result = adapter.get_topic(topic_hash)

        assert result is not None

    def test_get_topic_not_found(self, adapter):
        """Should return None for non-existent topic."""
        result = adapter.get_topic("nonexistent")

        assert result is None


class TestGetDebate:
    """Tests for get_debate method."""

    def test_get_debate_with_prefix(self, adapter, mock_scheduled_debate_record):
        """Should get debate with pl_debate_ prefix."""
        debate_id = adapter.store_scheduled_debate(mock_scheduled_debate_record)

        result = adapter.get_debate(debate_id)

        assert result is not None
        assert result["id"] == debate_id

    def test_get_debate_without_prefix(self, adapter, mock_scheduled_debate_record):
        """Should get debate without prefix."""
        debate_id = adapter.store_scheduled_debate(mock_scheduled_debate_record)
        original_id = mock_scheduled_debate_record.id

        result = adapter.get_debate(original_id)

        assert result is not None

    def test_get_debate_not_found(self, adapter):
        """Should return None for non-existent debate."""
        result = adapter.get_debate("nonexistent")

        assert result is None


# =============================================================================
# Search Past Debates Tests
# =============================================================================


class TestSearchPastDebates:
    """Tests for search_past_debates method."""

    def test_search_past_debates_exact_match(self, adapter):
        """Should find exact topic hash match."""
        # Store a debate with unix timestamp
        debate_data = {
            "id": "pl_debate_001",
            "topic_hash": hashlib.sha256(b"test topic").hexdigest()[:16],
            "topic_text": "test topic",
            "created_at": time.time(),  # Unix timestamp
            "platform": "hackernews",
        }
        adapter._debates["pl_debate_001"] = debate_data

        results = adapter.search_past_debates("test topic", hours=48)

        assert len(results) == 1
        assert results[0]["match_type"] == "exact"

    def test_search_past_debates_similar_match(self, adapter):
        """Should find debates with keyword overlap."""
        # Store a debate
        debate_data = {
            "id": "pl_debate_001",
            "topic_hash": "different_hash",
            "topic_text": "AI language models discussion",
            "created_at": time.time(),
            "platform": "hackernews",
        }
        adapter._debates["pl_debate_001"] = debate_data

        results = adapter.search_past_debates("AI models and language", hours=48)

        assert len(results) == 1
        assert results[0]["match_type"] == "similar"
        assert results[0]["overlap_count"] >= 2

    def test_search_past_debates_time_filter(self, adapter):
        """Should filter debates by time window."""
        # Store an old debate
        old_debate = {
            "id": "pl_debate_old",
            "topic_hash": hashlib.sha256(b"test topic").hexdigest()[:16],
            "topic_text": "test topic",
            "created_at": time.time() - (72 * 3600),  # 72 hours ago
            "platform": "hackernews",
        }
        adapter._debates["pl_debate_old"] = old_debate

        results = adapter.search_past_debates("test topic", hours=48)

        assert len(results) == 0  # Should not find old debate

    def test_search_past_debates_limit(self, adapter):
        """Should respect limit parameter."""
        # Store multiple debates
        for i in range(5):
            debate_data = {
                "id": f"pl_debate_{i}",
                "topic_hash": f"hash_{i}",
                "topic_text": "AI language models",
                "created_at": time.time() - (i * 60),  # Different times
                "platform": "hackernews",
            }
            adapter._debates[f"pl_debate_{i}"] = debate_data

        results = adapter.search_past_debates("AI models language", hours=48, limit=3)

        assert len(results) == 3

    def test_search_past_debates_sorted_by_time(self, adapter):
        """Should return debates sorted by created_at descending."""
        # Store debates at different times
        adapter._debates["pl_debate_old"] = {
            "id": "pl_debate_old",
            "topic_hash": "h1",
            "topic_text": "AI language models",
            "created_at": time.time() - 3600,  # 1 hour ago
            "platform": "hackernews",
        }
        adapter._debates["pl_debate_new"] = {
            "id": "pl_debate_new",
            "topic_hash": "h2",
            "topic_text": "AI language models",
            "created_at": time.time() - 60,  # 1 minute ago
            "platform": "hackernews",
        }

        results = adapter.search_past_debates("AI models language", hours=48)

        assert results[0]["id"] == "pl_debate_new"

    def test_search_past_debates_no_match(self, adapter):
        """Should return empty list for no matches."""
        adapter._debates["pl_debate_001"] = {
            "id": "pl_debate_001",
            "topic_hash": "hash",
            "topic_text": "completely different topic",
            "created_at": time.time(),
            "platform": "hackernews",
        }

        results = adapter.search_past_debates("AI language models", hours=48)

        assert len(results) == 0


# =============================================================================
# Platform/Category Topics Tests
# =============================================================================


class TestGetPlatformTopics:
    """Tests for get_platform_topics method."""

    def test_get_platform_topics(self, adapter, mock_trending_topic):
        """Should get topics from specific platform."""
        adapter.store_trending_topic(mock_trending_topic)

        results = adapter.get_platform_topics("hackernews")

        assert len(results) == 1
        assert results[0]["platform"] == "hackernews"

    def test_get_platform_topics_empty(self, adapter):
        """Should return empty list for unknown platform."""
        results = adapter.get_platform_topics("unknown_platform")

        assert len(results) == 0

    def test_get_platform_topics_limit(self, adapter):
        """Should respect limit parameter."""
        from aragora.knowledge.mound.adapters.pulse_adapter import PulseAdapter

        for i in range(10):
            topic = MagicMock()
            topic.platform = "hackernews"
            topic.topic = f"Topic {i}"
            topic.volume = 10000 + i * 1000
            topic.category = "tech"
            topic.raw_data = {}
            adapter.store_trending_topic(topic)

        results = adapter.get_platform_topics("hackernews", limit=5)

        assert len(results) == 5


class TestGetCategoryTopics:
    """Tests for get_category_topics method."""

    def test_get_category_topics(self, adapter, mock_trending_topic):
        """Should get topics from specific category."""
        adapter.store_trending_topic(mock_trending_topic)

        results = adapter.get_category_topics("tech")

        assert len(results) == 1
        assert results[0]["category"] == "tech"

    def test_get_category_topics_empty(self, adapter):
        """Should return empty list for unknown category."""
        results = adapter.get_category_topics("unknown_category")

        assert len(results) == 0

    def test_get_category_topics_limit(self, adapter):
        """Should respect limit parameter."""
        for i in range(10):
            topic = MagicMock()
            topic.platform = "hackernews"
            topic.topic = f"Topic {i}"
            topic.volume = 10000 + i * 1000
            topic.category = "tech"
            topic.raw_data = {}
            adapter.store_trending_topic(topic)

        results = adapter.get_category_topics("tech", limit=5)

        assert len(results) == 5


# =============================================================================
# Trending Patterns Tests
# =============================================================================


class TestGetTrendingPatterns:
    """Tests for get_trending_patterns method."""

    def test_get_trending_patterns(self, adapter):
        """Should find recurring keywords."""
        # Store multiple topics with common keywords
        for i in range(5):
            topic = MagicMock()
            topic.platform = "hackernews"
            topic.topic = f"AI language models version {i}"
            topic.volume = 10000
            topic.category = "tech"
            topic.raw_data = {}
            adapter.store_trending_topic(topic)

        patterns = adapter.get_trending_patterns(min_occurrences=3)

        # Should find patterns like "language", "models"
        keywords = [p["keyword"] for p in patterns]
        assert any(kw in keywords for kw in ["language", "models"])

    def test_get_trending_patterns_min_occurrences(self, adapter):
        """Should filter by minimum occurrences."""
        # Store topics with different keywords
        for i in range(3):
            topic = MagicMock()
            topic.platform = "hackernews"
            topic.topic = f"common keyword unique_{i}"
            topic.volume = 10000
            topic.category = "tech"
            topic.raw_data = {}
            adapter.store_trending_topic(topic)

        patterns = adapter.get_trending_patterns(min_occurrences=3)

        # "common" and "keyword" appear 3 times, unique_X only once each
        keywords = [p["keyword"] for p in patterns]
        assert "common" in keywords
        assert "keyword" in keywords

    def test_get_trending_patterns_limit(self, adapter):
        """Should respect limit parameter."""
        for i in range(20):
            topic = MagicMock()
            topic.platform = "hackernews"
            topic.topic = f"word{i % 5} another{i % 3} more{i % 4}"
            topic.volume = 10000
            topic.category = "tech"
            topic.raw_data = {}
            adapter.store_trending_topic(topic)

        patterns = adapter.get_trending_patterns(min_occurrences=2, limit=5)

        assert len(patterns) <= 5

    def test_get_trending_patterns_skips_short_words(self, adapter):
        """Should skip words shorter than 3 characters."""
        topic = MagicMock()
        topic.platform = "hackernews"
        topic.topic = "AI is an important topic to discuss"
        topic.volume = 10000
        topic.category = "tech"
        topic.raw_data = {}
        adapter.store_trending_topic(topic)

        patterns = adapter.get_trending_patterns(min_occurrences=1)

        keywords = [p["keyword"] for p in patterns]
        assert "ai" not in keywords
        assert "is" not in keywords
        assert "an" not in keywords
        assert "to" not in keywords

    def test_get_trending_patterns_sorted_by_count(self, adapter):
        """Should return patterns sorted by occurrence count."""
        # Create topics where "common" appears most often
        for i in range(5):
            topic = MagicMock()
            topic.platform = "hackernews"
            topic.topic = "common word"
            topic.volume = 10000
            topic.category = "tech"
            topic.raw_data = {}
            adapter.store_trending_topic(topic)

        for i in range(3):
            topic = MagicMock()
            topic.platform = "hackernews"
            topic.topic = "rare word"
            topic.volume = 10000
            topic.category = "tech"
            topic.raw_data = {}
            adapter.store_trending_topic(topic)

        patterns = adapter.get_trending_patterns(min_occurrences=2)

        # "word" appears 8 times, "common" 5 times, "rare" 3 times
        if len(patterns) >= 2:
            assert patterns[0]["occurrence_count"] >= patterns[1]["occurrence_count"]


# =============================================================================
# Outcome Analytics Tests
# =============================================================================


class TestGetOutcomeAnalytics:
    """Tests for get_outcome_analytics method."""

    def test_get_outcome_analytics_empty(self, adapter):
        """Should return zeros for empty outcomes."""
        analytics = adapter.get_outcome_analytics()

        assert analytics["total_debates"] == 0
        assert analytics["consensus_rate"] == 0.0
        assert analytics["avg_confidence"] == 0.0
        assert analytics["avg_rounds"] == 0.0

    def test_get_outcome_analytics_with_data(self, adapter, mock_trending_topic_outcome):
        """Should calculate analytics correctly."""
        adapter.store_outcome(mock_trending_topic_outcome)

        # Store another outcome without consensus
        outcome2 = MagicMock()
        outcome2.topic = "Another topic"
        outcome2.platform = "hackernews"
        outcome2.debate_id = "debate_003"
        outcome2.consensus_reached = False
        outcome2.confidence = 0.5
        outcome2.rounds_used = 5
        outcome2.timestamp = time.time()
        outcome2.category = "tech"
        outcome2.volume = 3000
        adapter.store_outcome(outcome2)

        analytics = adapter.get_outcome_analytics()

        assert analytics["total_debates"] == 2
        assert analytics["consensus_rate"] == 0.5  # 1 out of 2
        assert analytics["avg_confidence"] == pytest.approx(0.7, abs=0.01)  # (0.9 + 0.5) / 2
        assert analytics["avg_rounds"] == pytest.approx(4.5, abs=0.01)  # (4 + 5) / 2

    def test_get_outcome_analytics_platform_filter(self, adapter):
        """Should filter by platform."""
        outcome1 = MagicMock()
        outcome1.topic = "Topic 1"
        outcome1.platform = "hackernews"
        outcome1.debate_id = "d1"
        outcome1.consensus_reached = True
        outcome1.confidence = 0.8
        outcome1.rounds_used = 3
        outcome1.timestamp = time.time()
        outcome1.category = "tech"
        outcome1.volume = 1000
        adapter.store_outcome(outcome1)

        outcome2 = MagicMock()
        outcome2.topic = "Topic 2"
        outcome2.platform = "reddit"
        outcome2.debate_id = "d2"
        outcome2.consensus_reached = False
        outcome2.confidence = 0.4
        outcome2.rounds_used = 5
        outcome2.timestamp = time.time()
        outcome2.category = "tech"
        outcome2.volume = 2000
        adapter.store_outcome(outcome2)

        analytics = adapter.get_outcome_analytics(platform="hackernews")

        assert analytics["total_debates"] == 1
        assert analytics["consensus_rate"] == 1.0
        assert analytics["platform"] == "hackernews"

    def test_get_outcome_analytics_category_filter(self, adapter):
        """Should filter by category."""
        outcome1 = MagicMock()
        outcome1.topic = "Topic 1"
        outcome1.platform = "hackernews"
        outcome1.debate_id = "d1"
        outcome1.consensus_reached = True
        outcome1.confidence = 0.8
        outcome1.rounds_used = 3
        outcome1.timestamp = time.time()
        outcome1.category = "tech"
        outcome1.volume = 1000
        adapter.store_outcome(outcome1)

        outcome2 = MagicMock()
        outcome2.topic = "Topic 2"
        outcome2.platform = "hackernews"
        outcome2.debate_id = "d2"
        outcome2.consensus_reached = False
        outcome2.confidence = 0.4
        outcome2.rounds_used = 5
        outcome2.timestamp = time.time()
        outcome2.category = "science"
        outcome2.volume = 2000
        adapter.store_outcome(outcome2)

        analytics = adapter.get_outcome_analytics(category="tech")

        assert analytics["total_debates"] == 1
        assert analytics["category"] == "tech"


# =============================================================================
# KnowledgeItem Conversion Tests
# =============================================================================


class TestToKnowledgeItem:
    """Tests for to_knowledge_item method."""

    def test_to_knowledge_item_high_quality(self, adapter, mock_trending_topic):
        """Should convert high quality topic to KnowledgeItem."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)
        topic_data = adapter._topics[topic_id]

        item = adapter.to_knowledge_item(topic_data)

        from aragora.knowledge.unified.types import ConfidenceLevel, KnowledgeSource

        assert item.id == topic_id
        assert item.content == mock_trending_topic.topic
        assert item.source == KnowledgeSource.PULSE
        assert item.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]

    def test_to_knowledge_item_medium_quality(self, adapter):
        """Should assign MEDIUM confidence for medium quality."""
        topic_data = {
            "id": "pl_topic_test",
            "topic": "Test topic",
            "topic_hash": "hash123",
            "platform": "hackernews",
            "category": "tech",
            "volume": 1000,
            "quality_score": 0.65,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        item = adapter.to_knowledge_item(topic_data)

        from aragora.knowledge.unified.types import ConfidenceLevel

        assert item.confidence == ConfidenceLevel.MEDIUM

    def test_to_knowledge_item_low_quality(self, adapter):
        """Should assign LOW confidence for low quality."""
        topic_data = {
            "id": "pl_topic_test",
            "topic": "Test topic",
            "topic_hash": "hash123",
            "platform": "hackernews",
            "category": "tech",
            "volume": 100,
            "quality_score": 0.45,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        item = adapter.to_knowledge_item(topic_data)

        from aragora.knowledge.unified.types import ConfidenceLevel

        assert item.confidence == ConfidenceLevel.LOW

    def test_to_knowledge_item_unverified_quality(self, adapter):
        """Should assign UNVERIFIED confidence for very low quality."""
        topic_data = {
            "id": "pl_topic_test",
            "topic": "Test topic",
            "topic_hash": "hash123",
            "platform": "hackernews",
            "category": "tech",
            "volume": 10,
            "quality_score": 0.2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        item = adapter.to_knowledge_item(topic_data)

        from aragora.knowledge.unified.types import ConfidenceLevel

        assert item.confidence == ConfidenceLevel.UNVERIFIED

    def test_to_knowledge_item_metadata(self, adapter):
        """Should include metadata in KnowledgeItem."""
        topic_data = {
            "id": "pl_topic_test",
            "topic": "Test topic",
            "topic_hash": "hash123",
            "platform": "hackernews",
            "category": "tech",
            "volume": 5000,
            "quality_score": 0.8,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        item = adapter.to_knowledge_item(topic_data)

        assert item.metadata["platform"] == "hackernews"
        assert item.metadata["category"] == "tech"
        assert item.metadata["volume"] == 5000
        assert item.metadata["quality_score"] == 0.8

    def test_to_knowledge_item_handles_missing_created_at(self, adapter):
        """Should handle missing created_at."""
        topic_data = {
            "id": "pl_topic_test",
            "topic": "Test topic",
            "topic_hash": "hash123",
            "platform": "hackernews",
            "category": "tech",
            "volume": 5000,
            "quality_score": 0.8,
        }

        item = adapter.to_knowledge_item(topic_data)

        assert item.created_at is not None


# =============================================================================
# Stats Tests
# =============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_empty(self, adapter):
        """Should return zeros for empty adapter."""
        stats = adapter.get_stats()

        assert stats["total_topics"] == 0
        assert stats["total_debates"] == 0
        assert stats["total_outcomes"] == 0
        assert stats["platforms_tracked"] == 0
        assert stats["categories_tracked"] == 0

    def test_get_stats_with_data(
        self,
        adapter,
        mock_trending_topic,
        mock_scheduled_debate_record,
        mock_trending_topic_outcome,
    ):
        """Should return correct stats with data."""
        adapter.store_trending_topic(mock_trending_topic)
        adapter.store_scheduled_debate(mock_scheduled_debate_record)
        adapter.store_outcome(mock_trending_topic_outcome)

        stats = adapter.get_stats()

        assert stats["total_topics"] == 1
        assert stats["total_debates"] == 1
        assert stats["total_outcomes"] == 1
        assert stats["platforms_tracked"] >= 1
        assert stats["categories_tracked"] >= 1


# =============================================================================
# Reverse Flow Tests
# =============================================================================


class TestInitReverseFlowState:
    """Tests for _init_reverse_flow_state method."""

    def test_init_reverse_flow_state(self, adapter):
        """Should initialize reverse flow state."""
        adapter._init_reverse_flow_state()

        assert hasattr(adapter, "_outcome_history")
        assert hasattr(adapter, "_km_validations")
        assert hasattr(adapter, "_km_coverage_cache")
        assert hasattr(adapter, "_km_priority_adjustments")
        assert hasattr(adapter, "_km_threshold_updates")
        assert hasattr(adapter, "_adjusted_min_quality")
        assert hasattr(adapter, "_adjusted_category_bonuses")

    def test_init_reverse_flow_state_idempotent(self, adapter):
        """Should not reset existing state."""
        adapter._init_reverse_flow_state()
        adapter._outcome_history.append({"test": "data"})

        adapter._init_reverse_flow_state()

        assert len(adapter._outcome_history) == 1


class TestRecordOutcomeForKM:
    """Tests for record_outcome_for_km method."""

    def test_record_outcome_for_km(self, adapter):
        """Should record outcome for KM analysis."""
        adapter.record_outcome_for_km(
            topic_id="pl_topic_001",
            debate_id="debate_001",
            outcome_success=True,
            confidence=0.85,
            rounds_used=3,
            category="tech",
        )

        assert len(adapter._outcome_history) == 1
        outcome = adapter._outcome_history[0]
        assert outcome["topic_id"] == "pl_topic_001"
        assert outcome["outcome_success"] is True
        assert outcome["confidence"] == 0.85

    def test_record_multiple_outcomes(self, adapter):
        """Should record multiple outcomes."""
        for i in range(5):
            adapter.record_outcome_for_km(
                topic_id=f"topic_{i}",
                debate_id=f"debate_{i}",
                outcome_success=i % 2 == 0,
                confidence=0.5 + i * 0.1,
                rounds_used=i + 1,
            )

        assert len(adapter._outcome_history) == 5


class TestUpdateQualityThresholdsFromKM:
    """Tests for update_quality_thresholds_from_km method."""

    @pytest.mark.asyncio
    async def test_insufficient_data(self, adapter):
        """Should return insufficient data recommendation."""
        result = await adapter.update_quality_thresholds_from_km(
            km_items=[{"metadata": {}}],
            min_items=10,
        )

        assert result.recommendation == "insufficient_data"
        assert result.adjustments_made == 0

    @pytest.mark.asyncio
    async def test_lower_threshold_on_high_low_quality_success(self, adapter):
        """Should lower threshold when low quality topics succeed."""
        km_items = []
        for i in range(20):
            km_items.append(
                {
                    "metadata": {
                        "category": "tech",
                        "quality_score": 0.45,  # Low quality
                        "outcome_success": True,  # But succeeds
                    }
                }
            )

        result = await adapter.update_quality_thresholds_from_km(km_items, min_items=10)

        assert result.new_min_quality < result.old_min_quality

    @pytest.mark.asyncio
    async def test_raise_threshold_on_low_quality_failure(self, adapter):
        """Should raise threshold when low quality topics fail."""
        km_items = []
        for i in range(20):
            km_items.append(
                {
                    "metadata": {
                        "category": "tech",
                        "quality_score": 0.45,  # Low quality
                        "outcome_success": False,  # And fails
                    }
                }
            )

        result = await adapter.update_quality_thresholds_from_km(km_items, min_items=10)

        assert result.new_min_quality > result.old_min_quality

    @pytest.mark.asyncio
    async def test_adjust_category_bonuses(self, adapter):
        """Should adjust category bonuses based on success rates."""
        km_items = []
        # Science has high success rate
        for i in range(10):
            km_items.append(
                {
                    "metadata": {
                        "category": "science",
                        "quality_score": 0.6,
                        "outcome_success": True,
                    }
                }
            )
        # Entertainment has low success rate
        for i in range(10):
            km_items.append(
                {
                    "metadata": {
                        "category": "entertainment",
                        "quality_score": 0.6,
                        "outcome_success": False,
                    }
                }
            )

        result = await adapter.update_quality_thresholds_from_km(km_items, min_items=10)

        # Science bonus should increase, entertainment should decrease
        assert result.new_category_bonuses.get("science", 0) >= result.old_category_bonuses.get(
            "science", 0
        )


class TestGetKMTopicCoverage:
    """Tests for get_km_topic_coverage method."""

    @pytest.mark.asyncio
    async def test_no_km_items(self, adapter):
        """Should return low coverage for no items."""
        coverage = await adapter.get_km_topic_coverage("New topic", km_items=[])

        assert coverage.coverage_score == 0.0
        assert coverage.recommendation == "proceed"

    @pytest.mark.asyncio
    async def test_high_coverage_high_consensus(self, adapter):
        """Should recommend skip for well-covered topics with high consensus."""
        km_items = []
        for i in range(12):
            km_items.append(
                {
                    "metadata": {
                        "outcome_success": True,
                        "confidence": 0.85,
                        "rounds_used": 3,
                    }
                }
            )

        coverage = await adapter.get_km_topic_coverage("Well-covered topic", km_items)

        assert coverage.coverage_score > 0.8
        assert coverage.consensus_rate > 0.7
        assert coverage.recommendation == "skip"
        assert coverage.priority_adjustment < 0

    @pytest.mark.asyncio
    async def test_partial_coverage_low_consensus(self, adapter):
        """Should recommend proceed for partial coverage with low consensus."""
        km_items = []
        for i in range(6):
            km_items.append(
                {
                    "metadata": {
                        "outcome_success": i < 2,  # Only 2 out of 6 succeed
                        "confidence": 0.5,
                        "rounds_used": 5,
                    }
                }
            )

        coverage = await adapter.get_km_topic_coverage("Contested topic", km_items)

        assert coverage.coverage_score > 0.5
        assert coverage.consensus_rate < 0.5
        assert coverage.recommendation == "proceed"

    @pytest.mark.asyncio
    async def test_novel_topic(self, adapter):
        """Should recommend proceed with priority boost for novel topics."""
        km_items = [{"metadata": {"outcome_success": True}}]

        coverage = await adapter.get_km_topic_coverage("Novel topic", km_items)

        assert coverage.coverage_score < 0.2
        assert coverage.recommendation == "proceed"
        assert coverage.priority_adjustment > 0


class TestValidateTopicFromKM:
    """Tests for validate_topic_from_km method."""

    @pytest.mark.asyncio
    async def test_no_outcomes(self, adapter):
        """Should return keep recommendation for no outcomes."""
        validation = await adapter.validate_topic_from_km("pl_topic_001", km_cross_refs=[])

        assert validation.recommendation == "keep"
        assert validation.km_confidence == 0.5

    @pytest.mark.asyncio
    async def test_high_success_rate(self, adapter):
        """Should recommend boost for high success rate."""
        km_cross_refs = []
        for i in range(10):
            km_cross_refs.append(
                {
                    "metadata": {
                        "outcome_success": True,
                        "confidence": 0.9,
                        "rounds_used": 2,
                    }
                }
            )

        validation = await adapter.validate_topic_from_km("pl_topic_001", km_cross_refs)

        assert validation.outcome_success_rate >= 0.8
        assert validation.recommendation == "boost"
        assert validation.priority_adjustment > 0

    @pytest.mark.asyncio
    async def test_low_success_rate(self, adapter):
        """Should recommend demote for low success rate."""
        km_cross_refs = []
        for i in range(10):
            km_cross_refs.append(
                {
                    "metadata": {
                        "outcome_success": False,
                        "confidence": 0.3,
                        "rounds_used": 5,
                    }
                }
            )

        validation = await adapter.validate_topic_from_km("pl_topic_001", km_cross_refs)

        assert validation.outcome_success_rate < 0.3
        assert validation.recommendation == "demote"
        assert validation.priority_adjustment < 0

    @pytest.mark.asyncio
    async def test_combines_internal_and_km_outcomes(self, adapter):
        """Should combine internal outcomes with KM cross-refs."""
        # Record internal outcomes
        adapter.record_outcome_for_km(
            topic_id="pl_topic_001",
            debate_id="d1",
            outcome_success=True,
            confidence=0.8,
            rounds_used=3,
        )

        km_cross_refs = [
            {
                "metadata": {
                    "outcome_success": True,
                    "confidence": 0.9,
                    "rounds_used": 2,
                }
            }
        ]

        validation = await adapter.validate_topic_from_km("pl_topic_001", km_cross_refs)

        assert validation.similar_debates_count == 2


class TestApplySchedulingRecommendation:
    """Tests for apply_scheduling_recommendation method."""

    @pytest.mark.asyncio
    async def test_topic_not_found(self, adapter):
        """Should return not applied for missing topic."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicValidation

        validation = KMTopicValidation(
            topic_id="nonexistent",
            recommendation="boost",
            priority_adjustment=0.1,
        )

        result = await adapter.apply_scheduling_recommendation(validation)

        assert result.was_applied is False
        assert result.reason == "topic_not_found"

    @pytest.mark.asyncio
    async def test_no_change_recommendation(self, adapter, mock_trending_topic):
        """Should not apply changes for keep recommendation."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicValidation

        topic_id = adapter.store_trending_topic(mock_trending_topic)

        validation = KMTopicValidation(
            topic_id=topic_id,
            recommendation="keep",
            priority_adjustment=0.0,
        )

        result = await adapter.apply_scheduling_recommendation(validation)

        assert result.was_applied is False
        assert result.reason == "no_change"

    @pytest.mark.asyncio
    async def test_boost_recommendation(self, adapter, mock_trending_topic):
        """Should apply boost recommendation."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicValidation

        topic_id = adapter.store_trending_topic(mock_trending_topic)
        original_quality = adapter._topics[topic_id]["quality_score"]

        validation = KMTopicValidation(
            topic_id=topic_id,
            recommendation="boost",
            priority_adjustment=0.1,
            km_confidence=0.9,
        )

        result = await adapter.apply_scheduling_recommendation(validation)

        assert result.was_applied is True
        assert result.adjusted_priority > result.original_priority
        assert adapter._topics[topic_id]["km_validated"] is True

    @pytest.mark.asyncio
    async def test_demote_recommendation(self, adapter, mock_trending_topic):
        """Should apply demote recommendation."""
        from aragora.knowledge.mound.adapters.pulse_adapter import KMTopicValidation

        topic_id = adapter.store_trending_topic(mock_trending_topic)

        validation = KMTopicValidation(
            topic_id=topic_id,
            recommendation="demote",
            priority_adjustment=-0.1,
            km_confidence=0.9,
        )

        result = await adapter.apply_scheduling_recommendation(validation)

        assert result.was_applied is True
        assert result.adjusted_priority < result.original_priority


class TestSyncValidationsFromKM:
    """Tests for sync_validations_from_km method."""

    @pytest.mark.asyncio
    async def test_empty_items(self, adapter):
        """Should handle empty items list."""
        result = await adapter.sync_validations_from_km(km_items=[])

        assert result.topics_analyzed == 0
        assert result.topics_adjusted == 0

    @pytest.mark.asyncio
    async def test_sync_with_topics(self, adapter, mock_trending_topic):
        """Should sync validations for topics."""
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        km_items = []
        for i in range(15):
            km_items.append(
                {
                    "metadata": {
                        "topic_id": topic_id,
                        "category": "tech",
                        "quality_score": 0.7,
                        "outcome_success": True,
                        "confidence": 0.85,
                        "rounds_used": 3,
                    }
                }
            )

        result = await adapter.sync_validations_from_km(km_items)

        assert result.topics_analyzed >= 1
        assert result.duration_ms >= 0
        assert "total_items" in result.metadata

    @pytest.mark.asyncio
    async def test_sync_handles_errors(self, adapter):
        """Should handle errors gracefully."""
        # Create items that might cause errors
        km_items = [
            {
                "metadata": {
                    "topic_id": "nonexistent_topic",
                    "category": "tech",
                    "quality_score": 0.7,
                    "outcome_success": True,
                }
            }
        ]

        result = await adapter.sync_validations_from_km(km_items)

        # Should complete without raising
        assert isinstance(result.errors, list)


class TestGetReverseFlowStats:
    """Tests for get_reverse_flow_stats method."""

    def test_get_reverse_flow_stats_initial(self, adapter):
        """Should return initial stats."""
        stats = adapter.get_reverse_flow_stats()

        assert stats["outcome_history_count"] == 0
        assert stats["validations_stored"] == 0
        assert stats["coverage_cache_size"] == 0
        assert stats["km_priority_adjustments"] == 0
        assert stats["km_threshold_updates"] == 0
        assert stats["current_min_quality"] == 0.6

    def test_get_reverse_flow_stats_with_data(self, adapter):
        """Should return stats with recorded data."""
        adapter.record_outcome_for_km(
            topic_id="topic_1",
            debate_id="debate_1",
            outcome_success=True,
            confidence=0.8,
            rounds_used=3,
        )

        stats = adapter.get_reverse_flow_stats()

        assert stats["outcome_history_count"] == 1


class TestClearReverseFlowState:
    """Tests for clear_reverse_flow_state method."""

    def test_clear_reverse_flow_state(self, adapter):
        """Should clear all reverse flow state."""
        adapter.record_outcome_for_km(
            topic_id="topic_1",
            debate_id="debate_1",
            outcome_success=True,
            confidence=0.8,
            rounds_used=3,
        )

        adapter.clear_reverse_flow_state()

        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_count"] == 0
        assert stats["current_min_quality"] == 0.6


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_store_topic_with_special_characters(self, adapter):
        """Should handle topics with special characters."""
        topic = MagicMock()
        topic.platform = "hackernews"
        topic.topic = "AI & ML: What's next? [2024]"
        topic.volume = 10000
        topic.category = "tech"
        topic.raw_data = {}

        topic_id = adapter.store_trending_topic(topic)

        assert topic_id is not None

    def test_store_topic_with_unicode(self, adapter):
        """Should handle topics with unicode characters."""
        topic = MagicMock()
        topic.platform = "hackernews"
        topic.topic = "AI讨论 日本語テスト emoji 🚀"
        topic.volume = 10000
        topic.category = "tech"
        topic.raw_data = {}

        topic_id = adapter.store_trending_topic(topic)

        assert topic_id is not None

    def test_store_topic_with_empty_topic(self, adapter):
        """Should handle empty topic text."""
        topic = MagicMock()
        topic.platform = "hackernews"
        topic.topic = ""
        topic.volume = 10000
        topic.category = "tech"
        topic.raw_data = {}

        topic_id = adapter.store_trending_topic(topic)

        # Should still store (quality score depends on volume)
        if topic_id:
            assert adapter._topics[topic_id]["topic"] == ""

    def test_store_outcome_with_none_values(self, adapter):
        """Should handle outcomes with None values."""
        outcome = MagicMock()
        outcome.topic = "Test topic"
        outcome.platform = "hackernews"
        outcome.debate_id = "d1"
        outcome.consensus_reached = None
        outcome.confidence = None
        outcome.rounds_used = None
        outcome.timestamp = time.time()
        outcome.category = None
        outcome.volume = None

        outcome_id = adapter.store_outcome(outcome)

        assert outcome_id is not None

    def test_search_debates_with_empty_query(self, adapter):
        """Should handle empty search query."""
        results = adapter.search_past_debates("", hours=48)

        assert isinstance(results, list)

    def test_get_outcome_analytics_handles_none_values(self, adapter):
        """Should handle None values in outcomes."""
        adapter._outcomes["o1"] = {
            "consensus_reached": None,
            "confidence": None,
            "rounds_used": None,
            "platform": "hackernews",
            "category": "tech",
        }

        analytics = adapter.get_outcome_analytics()

        # Should not raise
        assert analytics["total_debates"] == 1

    def test_to_knowledge_item_handles_invalid_date(self, adapter):
        """Should handle invalid date format."""
        topic_data = {
            "id": "pl_topic_test",
            "topic": "Test topic",
            "topic_hash": "hash123",
            "platform": "hackernews",
            "category": "tech",
            "volume": 5000,
            "quality_score": 0.8,
            "created_at": "invalid_date",
        }

        item = adapter.to_knowledge_item(topic_data)

        # Should use current time as fallback
        assert item.created_at is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for PulseAdapter."""

    def test_full_topic_lifecycle(self, adapter, mock_trending_topic, mock_trending_topic_outcome):
        """Should handle full topic lifecycle."""
        # Store topic
        topic_id = adapter.store_trending_topic(mock_trending_topic)
        assert topic_id is not None

        # Get topic
        topic = adapter.get_topic(topic_id)
        assert topic is not None

        # Store outcome
        outcome_id = adapter.store_outcome(mock_trending_topic_outcome)
        assert outcome_id is not None

        # Get analytics
        analytics = adapter.get_outcome_analytics()
        assert analytics["total_debates"] == 1

        # Get stats
        stats = adapter.get_stats()
        assert stats["total_topics"] == 1
        assert stats["total_outcomes"] == 1

    @pytest.mark.asyncio
    async def test_full_km_reverse_flow(self, adapter, mock_trending_topic):
        """Should handle full KM reverse flow."""
        # Store topic
        topic_id = adapter.store_trending_topic(mock_trending_topic)

        # Record outcomes
        for i in range(15):
            adapter.record_outcome_for_km(
                topic_id=topic_id,
                debate_id=f"debate_{i}",
                outcome_success=True,
                confidence=0.85,
                rounds_used=3,
                category="tech",
            )

        # Create KM items
        km_items = []
        for i in range(15):
            km_items.append(
                {
                    "metadata": {
                        "topic_id": topic_id,
                        "category": "tech",
                        "quality_score": 0.7,
                        "outcome_success": True,
                        "confidence": 0.85,
                        "rounds_used": 3,
                    }
                }
            )

        # Sync validations
        result = await adapter.sync_validations_from_km(km_items)

        assert result.topics_analyzed >= 1

        # Check reverse flow stats
        stats = adapter.get_reverse_flow_stats()
        assert stats["outcome_history_count"] >= 15


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Should export all public classes."""
        from aragora.knowledge.mound.adapters.pulse_adapter import __all__

        expected = [
            "PulseAdapter",
            "TopicSearchResult",
            "KMQualityThresholdUpdate",
            "KMTopicCoverage",
            "KMSchedulingRecommendation",
            "KMTopicValidation",
            "PulseKMSyncResult",
        ]

        for name in expected:
            assert name in __all__

    def test_import_from_package(self):
        """Should be importable from adapters package."""
        from aragora.knowledge.mound.adapters.pulse_adapter import (
            PulseAdapter,
            TopicSearchResult,
            KMQualityThresholdUpdate,
            KMTopicCoverage,
            KMSchedulingRecommendation,
            KMTopicValidation,
            PulseKMSyncResult,
        )

        assert PulseAdapter is not None
        assert TopicSearchResult is not None
        assert KMQualityThresholdUpdate is not None
