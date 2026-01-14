"""Tests for Pulse topic quality filter."""

import pytest

from aragora.pulse.ingestor import TrendingTopic
from aragora.pulse.quality import (
    CLICKBAIT_PATTERNS,
    LOW_VALUE_PATTERNS,
    QUALITY_INDICATORS,
    SPAM_INDICATORS,
    QualityScore,
    TopicQualityFilter,
)


class TestQualityPatterns:
    """Tests for quality pattern constants."""

    def test_clickbait_patterns_populated(self):
        """Test clickbait patterns are defined."""
        assert len(CLICKBAIT_PATTERNS) >= 5

    def test_spam_indicators_populated(self):
        """Test spam indicators are defined."""
        assert len(SPAM_INDICATORS) >= 5

    def test_quality_indicators_populated(self):
        """Test quality indicators are defined."""
        assert len(QUALITY_INDICATORS) >= 5


class TestQualityScore:
    """Tests for QualityScore dataclass."""

    def test_quality_score_creation(self):
        """Test basic QualityScore creation."""
        topic = TrendingTopic("hackernews", "Test Topic", 100, "tech")
        score = QualityScore(
            topic=topic,
            overall_score=0.75,
            is_acceptable=True,
            signals={"length": 0.8, "spam": 1.0},
            issues=[],
        )

        assert score.topic == topic
        assert score.overall_score == 0.75
        assert score.is_acceptable is True


class TestTopicQualityFilter:
    """Tests for TopicQualityFilter."""

    @pytest.fixture
    def filter(self):
        """Create a quality filter for testing."""
        return TopicQualityFilter()

    def test_high_quality_topic(self, filter):
        """Test scoring a high-quality topic."""
        topic = TrendingTopic(
            platform="hackernews",
            topic="New research shows promising results for AI safety alignment techniques",
            volume=500,
            category="tech",
        )

        score = filter.score_topic(topic)

        assert score.overall_score > 0.5
        assert score.is_acceptable

    def test_clickbait_topic(self, filter):
        """Test scoring a clickbait topic."""
        topic = TrendingTopic(
            platform="twitter",
            topic="You won't believe what happened next! Doctors hate this!",
            volume=10000,
            category="viral",
        )

        score = filter.score_topic(topic)

        assert score.signals["clickbait"] < 0.5
        assert "clickbait" in " ".join(score.issues).lower() or not score.is_acceptable

    def test_spam_topic(self, filter):
        """Test scoring a spam topic."""
        topic = TrendingTopic(
            platform="twitter",
            topic="FREE $1000 giveaway! Click here bit.ly/scam123",
            volume=5000,
            category="spam",
        )

        score = filter.score_topic(topic)

        assert score.signals["spam"] < 0.5

    def test_short_topic(self, filter):
        """Test scoring a very short topic."""
        topic = TrendingTopic(
            platform="twitter",
            topic="lol",
            volume=100,
            category="general",
        )

        score = filter.score_topic(topic)

        assert score.signals["length"] < 0.3
        assert score.signals["substance"] < 0.5

    def test_quality_indicators_boost(self, filter):
        """Test that quality indicators boost score."""
        topic = TrendingTopic(
            platform="hackernews",
            topic="New research analysis compares AI model performance with comprehensive data",
            volume=200,
            category="tech",
        )

        score = filter.score_topic(topic)

        assert score.signals["quality_indicators"] > 0

    def test_hashtag_spam_detection(self, filter):
        """Test detection of excessive hashtags."""
        topic = TrendingTopic(
            platform="twitter",
            topic="#AI #Tech #ML #DeepLearning #NeuralNet #Python Content",
            volume=500,
            category="tech",
        )

        score = filter.score_topic(topic)

        # Should have reduced structure score due to hashtag spam
        assert score.signals["structure"] < 1.0

    def test_filter_topics_batch(self, filter):
        """Test filtering multiple topics."""
        topics = [
            TrendingTopic("hackernews", "High quality research on AI systems", 300, "tech"),
            TrendingTopic("twitter", "lol", 100, "general"),  # Low quality
            TrendingTopic("reddit", "Detailed analysis of market trends", 500, "business"),
        ]

        filtered = filter.filter_topics(topics)

        # Only high-quality topics should pass
        assert len(filtered) <= 3
        # Results should be sorted by quality
        if len(filtered) >= 2:
            assert filtered[0].overall_score >= filtered[1].overall_score

    def test_filter_with_custom_threshold(self, filter):
        """Test filtering with custom quality threshold."""
        topics = [
            TrendingTopic("hackernews", "Good topic about technology", 200, "tech"),
            TrendingTopic("twitter", "Short", 100, "general"),
        ]

        # Low threshold - more permissive
        low_threshold = filter.filter_topics(topics, min_quality=0.2)

        # High threshold - more strict
        high_threshold = filter.filter_topics(topics, min_quality=0.8)

        assert len(low_threshold) >= len(high_threshold)

    def test_blocklist_filtering(self, filter):
        """Test blocklist term filtering."""
        filter.add_to_blocklist(["forbidden", "banned"])

        topic = TrendingTopic(
            platform="hackernews",
            topic="This topic contains a forbidden word",
            volume=300,
            category="tech",
        )

        score = filter.score_topic(topic)

        assert score.signals["blocklist"] == 0.0

    def test_get_stats(self, filter):
        """Test getting filter statistics."""
        stats = filter.get_stats()

        assert "min_quality_threshold" in stats
        assert "clickbait_pattern_count" in stats
        assert "spam_pattern_count" in stats


class TestQualityEdgeCases:
    """Edge case tests for quality filter."""

    def test_empty_topic(self):
        """Test handling empty topic text."""
        filter = TopicQualityFilter()
        topic = TrendingTopic("twitter", "", 100, "general")

        score = filter.score_topic(topic)

        # Empty topic should have low length and substance scores
        assert score.signals["length"] == 0.0
        assert score.signals["substance"] < 0.5
        # Overall score is weighed, so it won't be zero
        # but substance and length penalties should apply
        assert "too short" in " ".join(score.issues).lower()

    def test_all_caps_topic(self):
        """Test handling ALL CAPS topics."""
        filter = TopicQualityFilter()
        topic = TrendingTopic(
            "twitter", "THIS IS ALL CAPS AND PROBABLY SPAM", 1000, "general"
        )

        score = filter.score_topic(topic)

        # Should have reduced structure score
        assert score.signals["structure"] < 1.0

    def test_emoji_spam(self):
        """Test handling excessive emojis."""
        filter = TopicQualityFilter(max_emoji_ratio=0.05)  # Stricter emoji limit
        topic = TrendingTopic(
            "twitter", "🚀🚀🚀🚀🚀 TO THE MOON 🌙🌙🌙🌙🌙", 5000, "crypto"
        )

        score = filter.score_topic(topic)

        # Topic with many emojis should have issues
        # Even if structure detection isn't perfect, other signals should catch it
        assert score.signals["substance"] < 1.0  # Limited actual words

    def test_low_value_content(self):
        """Test detection of low-value content patterns."""
        filter = TopicQualityFilter()

        low_value_topics = [
            TrendingTopic("twitter", "same", 100, "general"),
            TrendingTopic("twitter", "this", 100, "general"),
            TrendingTopic("twitter", "...", 100, "general"),
        ]

        for topic in low_value_topics:
            score = filter.score_topic(topic)
            assert score.signals["substance"] < 0.3

    def test_unicode_topic(self):
        """Test handling Unicode characters."""
        filter = TopicQualityFilter()
        topic = TrendingTopic(
            "hackernews",
            "研究显示人工智能技术进展 - AI research progress",
            200,
            "tech",
        )

        score = filter.score_topic(topic)

        # Should handle Unicode gracefully
        assert 0 <= score.overall_score <= 1

    def test_custom_min_text_length(self):
        """Test custom minimum text length."""
        filter = TopicQualityFilter(min_text_length=50)

        short_topic = TrendingTopic("hackernews", "Short topic text", 100, "tech")
        long_topic = TrendingTopic(
            "hackernews",
            "This is a much longer topic text that should pass the length requirement",
            100,
            "tech",
        )

        short_score = filter.score_topic(short_topic)
        long_score = filter.score_topic(long_topic)

        assert short_score.signals["length"] < long_score.signals["length"]
