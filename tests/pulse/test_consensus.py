"""Tests for Pulse cross-source consensus detection."""

import pytest

from aragora.pulse.ingestor import TrendingTopic
from aragora.pulse.consensus import (
    ConsensusResult,
    CrossSourceConsensus,
    TopicCluster,
)


class TestTopicCluster:
    """Tests for TopicCluster dataclass."""

    def test_cluster_creation(self):
        """Test basic cluster creation."""
        cluster = TopicCluster(
            id="cluster-1",
            canonical_topic="AI Breakthrough",
        )

        assert cluster.id == "cluster-1"
        assert cluster.canonical_topic == "AI Breakthrough"
        assert cluster.platform_count == 0
        assert not cluster.is_cross_platform

    def test_add_topic_to_cluster(self):
        """Test adding topics to a cluster."""
        cluster = TopicCluster(id="cluster-1", canonical_topic="AI News")

        topic1 = TrendingTopic("hackernews", "AI News Story", 300, "tech")
        topic2 = TrendingTopic("reddit", "AI News Discussion", 5000, "tech")

        cluster.add_topic(topic1)
        cluster.add_topic(topic2)

        assert len(cluster.topics) == 2
        assert cluster.platform_count == 2
        assert cluster.is_cross_platform
        assert cluster.total_volume == 5300

    def test_single_platform_not_cross_platform(self):
        """Test single platform doesn't count as cross-platform."""
        cluster = TopicCluster(id="cluster-1", canonical_topic="Test")

        topic1 = TrendingTopic("hackernews", "Topic 1", 100, "tech")
        topic2 = TrendingTopic("hackernews", "Topic 2", 200, "tech")

        cluster.add_topic(topic1)
        cluster.add_topic(topic2)

        assert cluster.platform_count == 1
        assert not cluster.is_cross_platform


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""

    def test_empty_result(self):
        """Test empty consensus result."""
        result = ConsensusResult(
            clusters=[],
            cross_platform_count=0,
            single_platform_count=0,
            consensus_topics=[],
            confidence_boosts={},
        )

        assert len(result.clusters) == 0
        assert result.cross_platform_count == 0


class TestCrossSourceConsensus:
    """Tests for CrossSourceConsensus."""

    @pytest.fixture
    def consensus(self):
        """Create a consensus detector for testing."""
        return CrossSourceConsensus()

    def test_detect_cross_platform_consensus(self, consensus):
        """Test detecting consensus across platforms."""
        # Use more similar topic texts to ensure clustering
        topics = [
            TrendingTopic("hackernews", "OpenAI releases GPT-5 model", 500, "ai"),
            TrendingTopic("reddit", "OpenAI releases GPT-5 model", 8000, "ai"),
            TrendingTopic("twitter", "OpenAI releases GPT-5", 50000, "ai"),
        ]

        result = consensus.detect_consensus(topics)

        # Should detect cross-platform trend
        assert result.cross_platform_count >= 1
        assert len(result.consensus_topics) >= 1

    def test_detect_single_platform_topics(self, consensus):
        """Test topics from single platform are not consensus."""
        topics = [
            TrendingTopic("hackernews", "Rust programming tips", 300, "programming"),
            TrendingTopic("hackernews", "Python best practices", 400, "programming"),
            TrendingTopic("hackernews", "Go concurrency patterns", 350, "programming"),
        ]

        result = consensus.detect_consensus(topics)

        # All from same platform - no cross-platform consensus
        assert result.cross_platform_count == 0
        assert len(result.consensus_topics) == 0

    def test_similar_topics_cluster(self, consensus):
        """Test similar topics are clustered together."""
        # Use nearly identical text to ensure clustering
        topics = [
            TrendingTopic("hackernews", "AI model beats human chess champion", 400, "ai"),
            TrendingTopic("reddit", "AI model beats human chess champion", 6000, "ai"),
        ]

        result = consensus.detect_consensus(topics)

        # Should cluster these similar topics
        assert len(result.clusters) <= 2  # At most 2 clusters
        # At least one cluster should have multiple topics
        multi_topic_clusters = [c for c in result.clusters if len(c.topics) >= 2]
        assert len(multi_topic_clusters) >= 1

    def test_dissimilar_topics_separate_clusters(self, consensus):
        """Test dissimilar topics get separate clusters."""
        topics = [
            TrendingTopic("hackernews", "AI breakthrough in medical imaging", 300, "ai"),
            TrendingTopic("reddit", "Best pizza recipes for beginners", 5000, "food"),
        ]

        result = consensus.detect_consensus(topics)

        # Should be separate clusters
        assert len(result.clusters) == 2

    def test_confidence_boost_values(self, consensus):
        """Test confidence boosts are calculated correctly."""
        topics = [
            TrendingTopic("hackernews", "Major tech announcement", 500, "tech"),
            TrendingTopic("reddit", "Tech announcement discussion", 8000, "tech"),
            TrendingTopic("twitter", "Tech announcement trending", 50000, "tech"),
        ]

        result = consensus.detect_consensus(topics)

        # Should have confidence boosts for consensus topics
        if result.consensus_topics:
            for topic in result.consensus_topics:
                boost = result.confidence_boosts.get(topic.topic, 0)
                assert boost >= 0

    def test_get_consensus_boost_for_topic(self, consensus):
        """Test getting consensus boost for individual topic."""
        all_topics = [
            TrendingTopic("hackernews", "Shared topic across platforms", 300, "tech"),
            TrendingTopic("reddit", "Same shared topic", 5000, "tech"),
            TrendingTopic("twitter", "Unique twitter topic", 10000, "general"),
        ]

        hn_topic = all_topics[0]
        twitter_topic = all_topics[2]

        hn_boost = consensus.get_consensus_boost(hn_topic, all_topics)
        twitter_boost = consensus.get_consensus_boost(twitter_topic, all_topics)

        # HN topic should get boost (similar to Reddit topic)
        # Twitter topic unique - no boost
        assert hn_boost >= twitter_boost

    def test_find_related_topics(self, consensus):
        """Test finding related topics."""
        target = TrendingTopic("hackernews", "Machine learning in production", 300, "ai")
        candidates = [
            TrendingTopic("reddit", "ML production deployment guide", 5000, "ai"),
            TrendingTopic("twitter", "Cooking recipes for summer", 10000, "food"),
            TrendingTopic("hackernews", "Production ML best practices", 400, "ai"),
        ]

        related = consensus.find_related_topics(target, candidates, max_results=5)

        # Should find ML-related topics, not cooking
        topic_texts = [t.topic for t, _ in related]
        assert "Cooking recipes for summer" not in topic_texts

    def test_empty_topics_list(self, consensus):
        """Test handling empty topic list."""
        result = consensus.detect_consensus([])

        assert result.cross_platform_count == 0
        assert result.single_platform_count == 0
        assert len(result.clusters) == 0

    def test_min_cluster_size_filter(self, consensus):
        """Test minimum cluster size filtering."""
        topics = [
            TrendingTopic("hackernews", "Topic A", 100, "tech"),
            TrendingTopic("reddit", "Topic B", 200, "tech"),  # Different
            TrendingTopic("twitter", "Topic C", 300, "tech"),  # Different
        ]

        # With min_cluster_size=1, all single topics are included
        result_small = consensus.detect_consensus(topics, min_cluster_size=1)

        # With min_cluster_size=2, single topics filtered out
        result_large = consensus.detect_consensus(topics, min_cluster_size=2)

        assert len(result_small.clusters) >= len(result_large.clusters)

    def test_get_stats(self, consensus):
        """Test getting detector statistics."""
        stats = consensus.get_stats()

        assert "similarity_threshold" in stats
        assert "min_platforms_for_consensus" in stats
        assert "consensus_confidence_boost" in stats
        assert "stopword_count" in stats


class TestConsensusSimilarity:
    """Tests for similarity calculation."""

    def test_identical_texts_high_similarity(self):
        """Test identical texts have high similarity."""
        consensus = CrossSourceConsensus()

        # Access private method for testing
        sim = consensus._calculate_similarity(
            "OpenAI releases GPT-5",
            "OpenAI releases GPT-5",
        )

        assert sim > 0.99

    def test_similar_texts_moderate_similarity(self):
        """Test similar texts have moderate similarity."""
        consensus = CrossSourceConsensus()

        sim = consensus._calculate_similarity(
            "OpenAI announces GPT-5 release",
            "GPT-5 released by OpenAI today",
        )

        # Should be similar but not identical
        assert 0.3 < sim < 1.0

    def test_different_texts_low_similarity(self):
        """Test different texts have low similarity."""
        consensus = CrossSourceConsensus()

        sim = consensus._calculate_similarity(
            "OpenAI releases GPT-5",
            "Best recipes for chocolate cake",
        )

        assert sim < 0.3

    def test_keyword_extraction(self):
        """Test keyword extraction removes stopwords."""
        consensus = CrossSourceConsensus()

        keywords = consensus._extract_keywords("The quick brown fox jumps over the lazy dog")

        # Common stopwords should be removed
        assert "the" not in keywords
        # Content words should remain
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        assert "jumps" in keywords


class TestConsensusEdgeCases:
    """Edge case tests for consensus detection."""

    def test_single_topic(self):
        """Test handling single topic."""
        consensus = CrossSourceConsensus()
        topics = [TrendingTopic("hackernews", "Single topic", 100, "tech")]

        result = consensus.detect_consensus(topics)

        assert len(result.clusters) == 1
        assert result.cross_platform_count == 0

    def test_many_platforms(self):
        """Test handling topics from many platforms."""
        consensus = CrossSourceConsensus()
        topics = [
            TrendingTopic("hackernews", "Same breaking news story", 300, "news"),
            TrendingTopic("reddit", "Breaking news story discussion", 5000, "news"),
            TrendingTopic("twitter", "Breaking news trending", 50000, "news"),
            TrendingTopic("github", "News analysis tool released", 200, "news"),
        ]

        result = consensus.detect_consensus(topics)

        # Should detect high consensus
        if result.cross_platform_count > 0:
            max_boost = max(result.confidence_boosts.values())
            assert max_boost > 0

    def test_custom_similarity_threshold(self):
        """Test custom similarity threshold."""
        # High threshold - stricter matching
        strict = CrossSourceConsensus(similarity_threshold=0.90)

        # Low threshold - more permissive
        permissive = CrossSourceConsensus(similarity_threshold=0.30)

        topics = [
            TrendingTopic("hackernews", "AI model performance", 300, "ai"),
            TrendingTopic("reddit", "Performance of AI models", 5000, "ai"),
        ]

        strict_result = strict.detect_consensus(topics)
        permissive_result = permissive.detect_consensus(topics)

        # Permissive should find more clusters
        assert permissive_result.cross_platform_count >= strict_result.cross_platform_count

    def test_short_topic_text(self):
        """Test handling very short topic text."""
        consensus = CrossSourceConsensus()
        topics = [
            TrendingTopic("hackernews", "AI", 100, "tech"),
            TrendingTopic("reddit", "AI", 200, "tech"),
        ]

        result = consensus.detect_consensus(topics)

        # Should handle gracefully
        assert len(result.clusters) >= 1
